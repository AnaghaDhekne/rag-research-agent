import streamlit as st
import mlflow
from mlflow.entities import SpanType
from databricks.sdk import WorkspaceClient
from openai import OpenAI

# ---------------------------------------------------------------------------
# MLflow tracing setup (Unity Catalog) — best-effort
# ---------------------------------------------------------------------------
# The app runs as a service principal that may not have access to the user's
# personal MLflow experiment.  We attempt to set up UC tracing; if it fails
# the app still works — only trace persistence is skipped.
MLFLOW_READY = False
_MLFLOW_ERR = ""
mlflow.set_tracking_uri("databricks")

# Unity Catalog trace storage requires a SQL warehouse to write trace
# spans to UC Delta tables.  This must be set BEFORE mlflow.set_experiment
# with a UC trace_location — without it, traces are created in memory but
# never persisted, which is why nothing appears in the experiment.
import os as _os
_os.environ.setdefault("MLFLOW_TRACING_SQL_WAREHOUSE_ID", "b87a856b0d313473")

# Try the user's UC-backed experiment first; fall back to a self-created
# workspace experiment that the service principal owns.
try:
    # NOTE: Do NOT pass trace_location here — the experiment already has
    # UC trace storage configured via tags (set by the notebook).  Passing
    # trace_location requires CAN_MANAGE on the experiment, but the app's
    # service principal only has CAN_EDIT.  set_experiment() without the
    # parameter only needs CAN_EDIT and still routes traces to UC.
    mlflow.set_experiment(
        "/Users/anaghagdhekne@gmail.com/experiment_1_uc",
    )
    MLFLOW_READY = True
except Exception as _uc_err:
    _MLFLOW_ERR = f"UC experiment failed: {_uc_err}"

# autolog is best-effort: it auto-traces OpenAI SDK calls inside traced
# functions, but the @mlflow.trace decorators work even without it.
# Keep it in its own try/except so a failure here does NOT cause the
# fallback below to switch the active experiment away from UC.
if MLFLOW_READY:
    try:
        mlflow.openai.autolog()
    except Exception:
        pass  # @mlflow.trace decorators still work without autolog

if not MLFLOW_READY:
    try:
        mlflow.set_experiment("research_assistant_agent")
        try:
            mlflow.openai.autolog()
        except Exception:
            pass
        MLFLOW_READY = True
    except Exception as e:
        MLFLOW_READY = False
        _MLFLOW_ERR = f"{str(e)}" if not _MLFLOW_ERR else f"{_MLFLOW_ERR} | Fallback also failed: {e}"

# ---------------------------------------------------------------------------
# Databricks clients
# ---------------------------------------------------------------------------
w = WorkspaceClient()

# ---------------------------------------------------------------------------
# Diagnostic logging — write startup status to a UC table for debugging
# ---------------------------------------------------------------------------
try:
    from databricks.sdk.service.sql import StatementParameterListItem as _SPI
    _diag_table = "main.default.research_assistant_diagnostics"
    w.statement_execution.execute_statement(
        statement=f"""CREATE TABLE IF NOT EXISTS {_diag_table} (
            ts TIMESTAMP NOT NULL,
            mlflow_ready BOOLEAN,
            mlflow_err STRING,
            mlflow_version STRING,
            experiment_name STRING,
            warehouse_env STRING
        )""",
        warehouse_id="b87a856b0d313473",
        wait_timeout="30s",
    )
    _exp_info = ""
    try:
        _exp = mlflow.get_experiment_by_name('/Users/anaghagdhekne@gmail.com/experiment_1_uc')
        _exp_info = f"ID={_exp.experiment_id}" if _exp else "NOT_FOUND"
    except Exception:
        _exp_info = "get_experiment_failed"
    w.statement_execution.execute_statement(
        statement=f"""INSERT INTO {_diag_table} VALUES (
            current_timestamp(),
            {str(MLFLOW_READY).lower()},
            :err,
            :ver,
            :exp,
            :wh
        )""",
        warehouse_id="b87a856b0d313473",
        wait_timeout="30s",
        parameters=[
            _SPI(name="err", value=str(_MLFLOW_ERR or "")),
            _SPI(name="ver", value=str(mlflow.__version__)),
            _SPI(name="exp", value=_exp_info),
            _SPI(name="wh", value=str(_os.environ.get('MLFLOW_TRACING_SQL_WAREHOUSE_ID', 'NOT_SET'))),
        ],
    )
except Exception:
    pass  # diagnostic only

# ---------------------------------------------------------------------------
# AI Gateway (Unity Gateway) — OpenAI-compatible API with fallback
# ---------------------------------------------------------------------------
# The app routes LLM traffic through the Unity Gateway for governance,
# logging, and rate limiting.  If the gateway is unavailable it falls back
# to the direct Foundation Model API via the Databricks SDK.
GATEWAY_BASE_URL = f"{w.config.host.rstrip('/')}/ai-gateway/mlflow/v1"
GATEWAY_MODEL = "system.ai.llama-4-maverick"
FALLBACK_MODEL = "databricks-meta-llama-3-3-70b-instruct"

GATEWAY_READY = False
try:
    _auth = w.config.authenticate()
    _token = _auth["Authorization"].replace("Bearer ", "")
    gateway_client = OpenAI(api_key=_token, base_url=GATEWAY_BASE_URL)
    # Quick smoke-test to confirm the gateway is reachable.
    gateway_client.chat.completions.create(
        model=GATEWAY_MODEL,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
    LLM_CLIENT = gateway_client
    LLM_MODEL = GATEWAY_MODEL
    GATEWAY_READY = True
except Exception as _e:
    GATEWAY_READY = False
    _GATEWAY_ERR = str(_e)

if not GATEWAY_READY:
    LLM_CLIENT = w.serving_endpoints.get_open_ai_client()
    LLM_MODEL = FALLBACK_MODEL

# ---------------------------------------------------------------------------
# Vector Search (Unity Catalog) — semantic retrieval with fallback
# ---------------------------------------------------------------------------
VS_ENDPOINT = "research-assistant-vs-endpoint"
VS_INDEX = "main.default.python_kb_docs_index"
VS_NUM_RESULTS = 3

VS_READY = False
try:
    _vs = w.vector_search_indexes.get_index(index_name=VS_INDEX)
    VS_READY = bool(_vs.status and _vs.status.ready)
except Exception:
    VS_READY = False

# ---------------------------------------------------------------------------
# Inference table logging (AI Gateway payload logging to Unity Catalog)
# ---------------------------------------------------------------------------
# Every LLM call is logged to a Delta table in Unity Catalog so you can
# audit requests, track token usage, and monitor latency — the same
# governance benefit as the AI Gateway's built-in inference tables.
INFERENCE_TABLE = "main.default.research_assistant_inference_logs"
WH_ID = "b87a856b0d313473"  # Serverless Starter Warehouse

INFERENCE_LOGGING_READY = False


def _ensure_inference_table():
    """Create the inference log table if it doesn't exist."""
    try:
        w.statement_execution.execute_statement(
            statement=f"""CREATE TABLE IF NOT EXISTS {INFERENCE_TABLE} (
                timestamp TIMESTAMP NOT NULL,
                request_id STRING NOT NULL,
                model STRING,
                gateway_used BOOLEAN,
                query STRING,
                retrieved_context STRING,
                response STRING,
                prompt_tokens INT,
                completion_tokens INT,
                total_tokens INT,
                latency_ms BIGINT
            )""",
            warehouse_id=WH_ID,
            wait_timeout="50s",
        )
        return True
    except Exception:
        return False


INFERENCE_LOGGING_READY = _ensure_inference_table()


def log_inference(model, gateway_used, query, retrieved_context, response,
                   prompt_tokens, completion_tokens, total_tokens, latency_ms):
    """Log an inference request/response to the Unity Catalog inference table."""
    if not INFERENCE_LOGGING_READY:
        return
    import uuid
    from databricks.sdk.service.sql import StatementParameterListItem

    request_id = str(uuid.uuid4())
    ctx = " ||| ".join(retrieved_context) if isinstance(retrieved_context, list) else str(retrieved_context or "")
    try:
        w.statement_execution.execute_statement(
            statement=f"""INSERT INTO {INFERENCE_TABLE} VALUES (
                current_timestamp(),
                :request_id,
                :model,
                :gateway_used,
                :query,
                :context,
                :response,
                :prompt_tokens,
                :completion_tokens,
                :total_tokens,
                :latency_ms
            )""",
            warehouse_id=WH_ID,
            wait_timeout="50s",
            parameters=[
                StatementParameterListItem(name="request_id", value=request_id),
                StatementParameterListItem(name="model", value=str(model or "")),
                StatementParameterListItem(name="gateway_used", value=str(gateway_used).lower(), type="BOOLEAN"),
                StatementParameterListItem(name="query", value=str(query or "")),
                StatementParameterListItem(name="context", value=ctx),
                StatementParameterListItem(name="response", value=str(response or "")),
                StatementParameterListItem(name="prompt_tokens", value=str(prompt_tokens or 0), type="INT"),
                StatementParameterListItem(name="completion_tokens", value=str(completion_tokens or 0), type="INT"),
                StatementParameterListItem(name="total_tokens", value=str(total_tokens or 0), type="INT"),
                StatementParameterListItem(name="latency_ms", value=str(latency_ms or 0), type="BIGINT"),
            ],
        )
    except Exception:
        pass  # logging is best-effort


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    {"topic": "Python Data Types", "content": "Python has several built-in data types: int (integers), float (floating-point numbers), str (strings), bool (booleans), list (ordered mutable sequences), tuple (ordered immutable sequences), set (unordered unique elements), dict (key-value mappings), and bytes. Use type() to check a value's type and isinstance() for type comparison. f-strings (f\"...\") provide formatted string interpolation."},
    {"topic": "Python Functions", "content": "Python functions are defined with the def keyword, followed by a name, parameters in parentheses, and a colon. They can have default arguments, *args (variable positional), **kwargs (variable keyword), keyword-only arguments (after *), and positional-only arguments (before /). Functions are first-class objects — they can be passed as arguments, returned from other functions, and assigned to variables."},
    {"topic": "Python Classes and OOP", "content": "Python supports object-oriented programming with classes defined using the class keyword. The __init__ method is the constructor. Instance methods take self as the first parameter. Class variables are shared across instances. Python supports inheritance (class Child(Parent):), multiple inheritance, and the super() function to call parent methods. Magic methods like __str__, __repr__, __eq__, __len__ customize object behavior."},
    {"topic": "Python Decorators", "content": "Decorators are functions that modify the behavior of other functions. They use the @decorator syntax above a function definition. A decorator takes a function as input and returns a new function. Use functools.wraps to preserve metadata of the decorated function. Decorators can accept arguments by using nested functions. Class-based decorators implement __init__ and __call__ methods."},
    {"topic": "Python Generators", "content": "Generators are functions that yield values lazily using the yield keyword instead of return. They produce items one at a time, making them memory-efficient for large sequences. Generator expressions use a syntax similar to list comprehensions but with parentheses: (x**2 for x in range(10)). The generator protocol uses __iter__ and __next__ methods. Use next() to get the next value and StopIteration signals completion."},
    {"topic": "Python Context Managers", "content": "Context managers handle resource setup and teardown using the with statement. The contextlib.contextmanager decorator turns a generator function into a context manager. Class-based context managers implement __enter__ (returns the resource) and __exit__ (handles cleanup, receives exception info). They ensure resources like files, locks, and database connections are properly released even if exceptions occur."},
    {"topic": "Python Exception Handling", "content": "Python uses try/except/else/finally blocks for exception handling. Multiple except clauses can catch different exception types. The else block runs if no exception occurred. The finally block always runs. Custom exceptions inherit from Exception. Use raise to throw exceptions. Python's exception hierarchy includes BaseException, Exception, ValueError, TypeError, KeyError, IndexError, AttributeError, RuntimeError, and others. Use except Exception as e to capture the exception object."},
    {"topic": "Python Comprehensions", "content": "List comprehensions create lists concisely: [expr for item in iterable if condition]. Dictionary comprehensions: {key: val for item in iterable}. Set comprehensions: {expr for item in iterable}. Generator expressions use parentheses instead of brackets. Comprehensions support nested loops and multiple if conditions. They are generally faster than equivalent for-loop code because they are optimized at the C level in CPython."},
    {"topic": "Python Async Await", "content": "Python's asyncio library provides async/await syntax for concurrent programming. Async functions (coroutines) are defined with async def and await is used to pause execution until an awaitable completes. asyncio.run() executes a coroutine from synchronous code. asyncio.gather() runs multiple coroutines concurrently. asyncio.create_task() schedules coroutines. The asyncio event loop manages execution. aiohttp and httpx provide async HTTP clients."},
    {"topic": "Python Type Hints", "content": "Python type hints (PEP 484) add static type information using the typing module. Basic hints: int, str, float, bool, list[int], dict[str, int], tuple[int, ...]. Optional[str] or str | None for nullable types. Callable[[int, str], bool] for function signatures. Use TypedDict for dict structures, @dataclass for lightweight classes. The mypy, pyright, and pyrefly type checkers validate hints statically. Type hints are optional and ignored at runtime."},
    {"topic": "Python Dataclasses", "content": "Dataclasses (PEP 557) simplify creating data-holding classes with the @dataclass decorator. It auto-generates __init__, __repr__, and __eq__ methods. Fields with default values must come after fields without defaults. Use field(default_factory=list) for mutable defaults. frozen=True makes instances immutable and hashable. slots=True (Python 3.10+) reduces memory usage. Dataclasses support inheritance and can include methods like regular classes."},
    {"topic": "Python Standard Library", "content": "Python's standard library includes: os (OS interfaces), sys (system-specific parameters), pathlib (path manipulation), collections (namedtuple, defaultdict, Counter, deque), itertools (chain, product, combinations), functools (lru_cache, partial, reduce), json (JSON encoding/decoding), re (regular expressions), datetime (date and time), logging (logging framework), typing (type hints), unittest and pytest (testing), multiprocessing and threading (concurrency), and socket (networking)."},
    {"topic": "Python File I/O", "content": "Python file I/O uses the built-in open() function with modes like 'r' (read), 'w' (write), 'a' (append), 'b' (binary), and '+' (read+write). Always use the with statement to ensure files are properly closed. The pathlib module provides object-oriented path manipulation (Path('/tmp/file.txt').read_text()). For large files, read line-by-line or in chunks. Use csv module for CSV files, pickle for serializing objects, and struct for binary data."},
    {"topic": "Python Regular Expressions", "content": "Python's re module provides regular expression operations. Key functions: re.match() (match at start), re.search() (search anywhere), re.findall() (all matches), re.sub() (substitute), re.split() (split by pattern). Use raw strings (r'...') for patterns. Groups use parentheses; named groups use (?P<name>...). re.compile() precompiles a pattern for reuse. The | operator provides alternation. Use re.IGNORECASE, re.MULTILINE, re.DOTALL flags."},
    {"topic": "Python Virtual Environments and pip", "content": "Python virtual environments isolate project dependencies. Create with: python -m venv .venv. Activate with: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows). pip install installs packages, pip freeze > requirements.txt exports dependencies, pip install -r requirements.txt reinstalls them. Use pip install --upgrade to update. Use uv or poetry for faster dependency management. Never install packages globally — always use a virtual environment."},
    {"topic": "Python Testing", "content": "Python testing uses unittest (built-in) or pytest (third-party). unittest requires test classes inheriting from unittest.TestCase with methods starting with test_. Use assertEqual, assertTrue, assertRaises for assertions. pytest uses plain assert statements and supports fixtures, parametrize, and markers. Run tests with: python -m unittest discover or pytest. Use unittest.mock for mocking. Coverage measurement with coverage.py or pytest-cov. Conftest.py provides shared pytest fixtures."},
    {"topic": "Python Logging", "content": "Python's logging module provides leveled logging: DEBUG, INFO, WARNING, ERROR, CRITICAL. Configure with logging.basicConfig(level=..., format=..., filename=...). Get a logger with logging.getLogger(__name__). Use logger.debug(), logger.info(), logger.warning(), logger.error(), logger.critical(). Handlers (StreamHandler, FileHandler) route log messages. Formatters control output layout. Use logging.config.dictConfig() for complex setups. Loggers form a hierarchy by dot-separated names."},
    {"topic": "Python Multithreading and Multiprocessing", "content": "Python's threading module runs threads sharing the same memory (limited by the GIL for CPU-bound tasks). multiprocessing runs separate processes with independent memory, bypassing the GIL. Use concurrent.futures.ThreadPoolExecutor and ProcessPoolExecutor for simple parallelism. For CPU-bound work, use multiprocessing; for I/O-bound work, use threading or asyncio. Use Queue for thread-safe communication, Lock and RLock for mutual exclusion, and Event for signaling."},
    {"topic": "Python Lambda Functions", "content": "Lambda functions are anonymous, single-expression functions defined with the lambda keyword: lambda x, y: x + y. They are limited to one expression and cannot contain statements. Commonly used with map(), filter(), sorted(key=...), and functools.reduce(). For complex logic, use a regular def function instead. Lambdas capture variables by reference (late binding) in closures — use default arguments to capture values at definition time."},
    {"topic": "Python Iterators and Iterables", "content": "Iterables are objects that can be iterated over (implement __iter__). Iterators are objects that produce values on demand (implement __iter__ and __next__). The iter() function gets an iterator from an iterable. next() returns the next value, raising StopIteration when exhausted. Generator functions are a convenient way to create iterators. The itertools module provides powerful iterator combinators: chain, count, cycle, repeat, islice, starmap, tee, groupby."},
    {"topic": "Python Property Decorator", "content": "The @property decorator turns a method into a computed attribute. Use @property for the getter, @<name>.setter for the setter, and @<name>.deleter for the deleter. Properties enable encapsulation — you can validate values on set, compute values on get, or change internal implementation without breaking the public API. They are an alternative to plain attributes when you need control over access. Use @property without a setter for read-only attributes."},
    {"topic": "Python Metaclasses", "content": "Metaclasses are classes that create classes. The default metaclass is type. A metaclass is defined by subclassing type and overriding __new__ or __init__. Metaclasses can customize class creation: enforce interfaces, register classes, modify attributes, or inject methods. Most code does not need metaclasses — use __init_subclass__ or class decorators for simpler metaprogramming. The __class_getitem__ method enables class subscripting (e.g., list[int])."},
    {"topic": "Python Packaging and Modules", "content": "Python packages are directories with __init__.py files. Modules are .py files. Use import statement to load modules; from ... import for specific names. The __name__ variable is '__main__' when run directly. Build packages with pyproject.toml (modern) or setup.py (legacy). Use pip install -e . for editable installs. The __all__ list controls what is exported with from package import *. Relative imports use leading dots (from . import module). Use importlib.import_module() for dynamic imports."},
    {"topic": "Python String Formatting", "content": "Python offers multiple string formatting methods: f-strings (f'Hello {name}') — Python 3.6+, preferred. str.format() — 'Hello {}'.format(name). %-formatting — 'Hello %s' % name (legacy). Template strings — string.Template('$name').substitute(name=...). For padding and alignment: f'{x:>10}' (right), f'{x:<10}' (left), f'{x:^10}' (center). For numbers: f'{x:.2f}' (2 decimals), f'{x:,}' (thousands separator), f'{x:08x}' (hex with zero-padding)."},
    {"topic": "Python Collections Module", "content": "The collections module provides specialized container datatypes: namedtuple (tuple with named fields), deque (double-ended queue, O(1) append/pop from both ends), Counter (dict for counting hashable objects, .most_common(n)), OrderedDict (dict preserving insertion order — standard dicts do this since 3.7), defaultdict (dict with default factory for missing keys), ChainMap (view of multiple dicts as one). The abc submodule provides abstract base classes like Iterable, Sequence, Mapping."},
    {"topic": "Python Enums", "content": "Python enums are created by subclassing enum.Enum. Each member is a constant with a name and value. Use @unique decorator to prevent duplicate values. IntEnum members are also ints; StrEnum members are also strings (Python 3.11+). Enums support iteration (for member in Color), membership testing (Color.RED in Color), and value lookup (Color('red')). Use enum.auto() for automatic value assignment. Enums are singletons — the same member object is always returned."},
    {"topic": "Python Walrus Operator", "content": "The walrus operator (:=), introduced in Python 3.8 (PEP 572), assigns a value as part of an expression. It enables assigning variables inside if conditions, while loops, and list comprehensions. Example: if (n := len(data)) > 10: print(f'Too long: {n}'). It avoids redundant function calls: while (line := file.readline()) != ''. Useful in comprehensions to avoid computing a value twice: [y := f(x), y**2, y**3]."},
    {"topic": "Python Pathlib", "content": "The pathlib module (Python 3.4+) provides object-oriented path manipulation. Use Path('/tmp') / 'file.txt' for joining paths. Methods: .exists(), .is_file(), .is_dir(), .mkdir(), .rmdir(), .unlink(), .rename(), .read_text(), .write_text(), .glob('*.txt'), .rglob('**/*.py'). Properties: .parent, .name, .stem, .suffix, .parts. Use Path.home() for home directory, Path.cwd() for current directory. Prefer pathlib over os.path for new code."},
    {"topic": "Python JSON Handling", "content": "The json module handles JSON encoding and decoding. json.dumps(obj) serializes a Python object to a JSON string. json.loads(s) deserializes a JSON string to a Python object. json.dump(obj, f) and json.load(f) work with file objects. Use indent=2 for pretty-printing. Custom serialization: subclass json.JSONEncoder and override default(). Use the json module's default keyword for handling dates and other non-serializable types. For performance, use orjson or ujson."},
    {"topic": "Python Slots", "content": "The __slots__ class attribute restricts instances to a fixed set of attributes, preventing dynamic attribute creation. It reduces memory usage and slightly improves attribute access speed. Define as __slots__ = ('x', 'y', 'z'). Slots prevent __dict__ creation, so you cannot add attributes not listed in __slots__. Inheritance: each subclass must define its own __slots__. If a subclass doesn't define __slots__, it gets a __dict__ anyway. With slots, instances are not weakly referenceable unless '__weakref__' is in __slots__."},
    {"topic": "Python f-string Debugging", "content": "Python 3.8+ added the = specifier to f-strings for debugging: f'{x=}' prints 'x=42'. Use f'{x=:>10}' to combine with format specs. f'{expr!r}' uses repr() instead of str(). For multiline: f'''...'''. F-strings can call functions: f'{len(data)}'. In Python 3.12+, f-strings can contain quotes matching the outer delimiter: f\"{dict['key']}\". F-strings are evaluated at runtime and can reference any in-scope variable."},
    {"topic": "Python Match Statement", "content": "Python 3.10+ introduced structural pattern matching with the match/case statement (PEP 634). It supports literal patterns (case 200:), capture patterns (case [x, y]:), wildcard (case _:), class patterns (case Point(x=0, y=0):), sequence patterns (case [1, 2, *rest]:), and mapping patterns (case {'status': 'ok', **rest}:). Guards use case x if x > 0:. The match statement is not a switch — it does not fall through. Use | for OR patterns: case 1 | 2 | 3:."},
    {"topic": "Python Garbage Collection", "content": "Python uses reference counting for memory management — objects are freed when their reference count reaches zero. A cyclic garbage collector handles reference cycles. The gc module controls the collector: gc.collect() triggers a collection, gc.disable() disables it. Use weakref module for weak references that don't prevent garbage collection. Context managers (with statement) and __del__ methods help with cleanup. For C extensions, use Py_INCREF/Py_DECREF. Memory is rarely a concern in pure Python, but large objects should be released explicitly."},
    {"topic": "Python Coroutines and Yield", "content": "Coroutines are functions that can pause and resume execution. The yield keyword makes a function a generator coroutine. Use .send(value) to send data into a coroutine. Use .throw(type) to throw an exception inside. Use .close() to terminate a coroutine. The @asyncio.coroutine decorator (deprecated) used yield from for delegation. Modern coroutines use async def and await. yield from delegates to a subgenerator. The yield from expr syntax is equivalent to a for loop yielding each value."},
    {"topic": "Python Dunder Methods", "content": "Dunder (double underscore) methods customize class behavior: __init__ (constructor), __str__ (user string), __repr__ (developer string), __len__ (len()), __getitem__ (x[i]), __setitem__ (x[i] = v), __iter__ (for x in obj), __contains__ (in operator), __call__ (obj()), __eq__/__lt__/__gt__ (comparisons), __hash__ (hash()), __enter__/__exit__ (context manager), __add__/__sub__/__mul__ (arithmetic operators), __bool__ (truthiness). Implement __repr__ before __str__ — __repr__ is the fallback for __str__."},
]

# ---------------------------------------------------------------------------
# Agent functions (traced with MLflow)
# ---------------------------------------------------------------------------
@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_context(query: str) -> list[str]:
    """Retrieve relevant documents via Vector Search, falling back to keywords."""
    # --- Vector Search (semantic) ---
    if VS_READY:
        try:
            results = w.vector_search_indexes.query_index(
                index_name=VS_INDEX,
                columns=["id", "topic", "content"],
                query_text=query,
                num_results=VS_NUM_RESULTS,
            )
            if results.result and results.result.data_array:
                return [
                    f"[{row[1]}] {row[2]}"
                    for row in results.result.data_array
                ]
        except Exception:
            pass  # fall through to keyword matching

    # --- Keyword fallback ---
    query_lower = query.lower()
    retrieved = []
    for doc in KNOWLEDGE_BASE:
        if doc["topic"].lower() in query_lower or any(
            word in query_lower for word in doc["topic"].lower().split()
        ):
            retrieved.append(f"[{doc['topic']}] {doc['content']}")
    return retrieved


@mlflow.trace(span_type=SpanType.LLM)
def generate_response(query: str, context: list[str], history: str = "") -> str:
    """Generate a response using the Databricks Foundation Model API."""
    context_text = "\n\n".join(context)
    if not context:
        return "I don't have information about that topic in my knowledge base. I can help with questions about Python data types, functions, classes and OOP, decorators, generators, context managers, exception handling, comprehensions, async/await, type hints, dataclasses, the standard library, file I/O, regular expressions, virtual environments, testing, logging, multithreading, lambda functions, iterators, properties, metaclasses, packaging, string formatting, collections, enums, the walrus operator, pathlib, JSON handling, slots, f-string debugging, match statements, garbage collection, coroutines, and dunder methods."
    prompt = (
        "You are a helpful Python programming assistant. Answer the question using ONLY the context provided below. "
        "If the context does not contain information relevant to the question, say: "
        "'I don\'t have information about that topic in my knowledge base.' "
        "Do not use your own knowledge to answer."
        f"\n\nContext:\n{context_text}"
        f"{history}"
        f"\n\nQuestion: {query}\n\nAnswer:"
    )
    import time
    _start = time.time()
    response = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    _latency_ms = int((time.time() - _start) * 1000)
    _content = response.choices[0].message.content

    # Log to inference table
    _usage = response.usage
    log_inference(
        model=LLM_MODEL,
        gateway_used=GATEWAY_READY,
        query=query,
        retrieved_context=context,
        response=_content,
        prompt_tokens=_usage.prompt_tokens if _usage else None,
        completion_tokens=_usage.completion_tokens if _usage else None,
        total_tokens=_usage.total_tokens if _usage else None,
        latency_ms=_latency_ms,
    )

    return _content


@mlflow.trace(span_type=SpanType.CHAIN)
def chat_agent(message: str, conversation_history: list[tuple[str, str]]) -> str:
    """Multi-turn chat agent with conversation memory and MLflow tracing."""
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": "streamlit_user",
            "mlflow.trace.session": "streamlit_session",
        }
    )
    context = retrieve_context(message)
    history_context = ""
    if conversation_history:
        history_lines = []
        for prev_msg, prev_resp in conversation_history[-5:]:
            history_lines.append(f"User: {prev_msg}\nAssistant: {prev_resp}")
        history_context = "\n\nPrevious conversation:\n" + "\n".join(history_lines)
    answer = generate_response(message, context, history_context)
    return answer


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Python Research Assistant", page_icon="🐍", layout="wide")

st.title("Python Research Assistant Agent")
st.markdown("Ask questions about Python functions, classes, decorators, generators, async/await, type hints, and more.")

with st.sidebar:
    st.header("About")
    st.markdown(f"**Model:** `{LLM_MODEL}`")
    if GATEWAY_READY:
        st.markdown("**LLM:**  AI Gateway (Unity Gateway)")
    else:
        st.markdown("**LLM:** ⚠️ Direct Foundation Model API (gateway unavailable)")
        with st.expander("Gateway error"):
            st.code(_GATEWAY_ERR[:500])
    if VS_READY:
        st.markdown("**Retrieval:**  Vector Search")
    else:
        st.markdown("**Retrieval:** ⚠️ Keyword matching (VS index not ready)")
    if INFERENCE_LOGGING_READY:
        st.markdown(f"**Inference logging:**  `{INFERENCE_TABLE}`")
    else:
        st.markdown("**Inference logging:** ⚠️ Disabled (warehouse unavailable)")
    st.markdown(f"**Knowledge base:** {len(KNOWLEDGE_BASE)} topics")
    st.markdown("**Topics:**")
    for doc in KNOWLEDGE_BASE:
        st.markdown(f"- {doc['topic']}")
    if MLFLOW_READY:
        st.markdown("**Tracing:**  MLflow → Unity Catalog")
    else:
        st.markdown("**Tracing:** ⚠️ Disabled")
        with st.expander("Details"):
            st.code(_MLFLOW_ERR[:500])
    st.markdown("---")
    st.caption("Built with Streamlit on Databricks Apps.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Python functions, classes, decorators, async/await…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    conversation_history = [
        (m["content"], n["content"])
        for m, n in zip(
            [m for m in st.session_state.messages if m["role"] == "user"],
            [m for m in st.session_state.messages if m["role"] == "assistant"],
        )
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = chat_agent(prompt, conversation_history)
            except Exception as e:
                reply = f"⚠️ Error: {e}"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
