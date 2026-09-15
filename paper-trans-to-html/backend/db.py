"""模块六 · MongoDB 连接层

数据默认存到 MongoDB（连接参数直接读 compose 用的那份 `.env`，不重复填一遍）。
**参数文件按顺序取第一个存在的**：`mongo_configuration/.env` → `docker/.env` → 仓库根 `.env`；
也可以用环境变量 `MONGO_ENV_FILE=/path/to/.env` 插到最前面（目录改名/换位置都不影响）。
配置读取顺序（高 → 低）：

| 顺序 | 来源 | 说明 |
| --- | --- | --- |
| 1 | 环境变量 `MONGO_URI` | 完整连接串，优先级最高（含云端 Atlas / 已有实例） |
| 2 | 环境变量 `MONGO_HOST/PORT/USER/PASSWORD/DB/AUTH_DB` | 拆开配 |
| 3 | 上面那份 `.env` | `docker compose` 用的那份，键名 `MONGO_PORT`/`MONGO_APP_USER`/`MONGO_APP_PASSWORD`/`MONGO_APP_DB` |
| 4 | 内置默认 | `127.0.0.1:27017`、账号 `paper/paper123`、库 `paper`（与 compose 的默认值一致） |

存储后端由 `APP_STORAGE` 决定：

- `auto`（默认）：能连上 MongoDB 就用 MongoDB，连不上**回退本地 JSON**（`data/*.json`），
  并在 `/api/config` 与前端提示里说明原因，避免服务直接不可用；
- `mongo`：强制用 MongoDB，连不上就报错（想“必须用库”时用它）；
- `json`：强制用本地 JSON（不需要 MongoDB 的轻量场景）。

连接是**懒加载 + 缓存**的：连不上时每次调用会快速失败（默认 1.5s 超时），
`refresh()` 可以在容器起来后重新探测（配合 `/api/storage/reconnect`）。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from urllib.parse import quote_plus

BASE_DIR = Path(__file__).resolve().parent.parent
# 连接参数文件的候选位置：按顺序取**第一个存在的**，也可用环境变量 MONGO_ENV_FILE 显式指定。
# （之前写死成单一路径，compose 目录一改名/换位置就读不到账号密码，只能悄悄回退默认值。）
ENV_FILE_CANDIDATES = (
    BASE_DIR.parent / "mongo_configuration" / ".env",      # 约定：MongoDB 自己那份 compose
    BASE_DIR / "docker" / ".env",          # 应用容器 compose（常把 MONGO_* 抄一份在这）
    BASE_DIR / ".env",                     # 项目根目录
)
# 参数文件**不再固定成某一个**：`db.MONGO_ENV_FILE` 与 `env_file_path()` 每次访问都按
# “第一个存在的”解析（见下方 env_file_path / __getattr__），这样目录改名、`.env` 晚一点
# 才创建都能跟上，而不是导入时就把值钉死在 ENV_FILE_CANDIDATES[0] 上。
DEFAULT_TIMEOUT_MS = 1500      # 连不上时快速失败，别把请求挂住

DEFAULTS = {
    "host": "127.0.0.1",
    "port": "27017",
    "user": "paper",
    "password": "paper123",
    "auth_db": "paper",
    "db": "paper",
}


class StorageUnavailable(RuntimeError):
    """MongoDB 不可用（未启动 / 认证失败 / 网络不通）。"""


def short_error(exc: Exception | str, limit: int = 180) -> str:
    """把 pymongo 那一大串 TopologyDescription 截成人能读的一行。"""
    text = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)
    first = text.splitlines()[0]
    for cut in (" (configured timeouts", ", Topology Description"):
        if cut in first:
            first = first.split(cut)[0]
    return first[:limit].strip()


# “账号/口令/authSource 不对”这一类报错的识别词（pymongo 在不同场景下措辞不同）
AUTH_ERROR_KEYS = ("authentication failed", "authenticationfailed", "usernotfound",
                   "bad auth", "unauthorized", "no such user")


def is_auth_error(err: str | None) -> bool:
    """这句报错是不是“认证问题”（而不是网络不通/库没起来）。"""
    text = (err or "").lower()
    return any(k in text for k in AUTH_ERROR_KEYS)


def auth_hint(err: str | None = None, cfg: dict | None = None) -> str:
    """认证失败时给一段能照做的排查步骤；不是认证问题就返回空串。

    这类报错最容易踩的坑：MongoDB 的业务账号建在**业务库**里（authSource = 那个库），
    而初始化脚本只在**数据卷为空**时执行 —— 卷是旧的就会被跳过，账号根本没建出来，
    表现就是“连得上库但 Authentication failed”。
    """
    if not is_auth_error(err):
        return ""
    who = f"{cfg.get('user')}@{cfg.get('auth_db')}" if cfg else ""
    return (f"这是认证失败（当前用 {who}）：数据库账号建在**业务库**里，authSource 必须是那个库名；"
            "账号没建出来 / 口令不符都会报 Authentication failed。"
            "① 核对 .env 的 MONGO_APP_USER / MONGO_APP_PASSWORD 与容器里的一致；"
            "② 初始化脚本只在**空数据卷**时执行（init/*.sh），卷是旧的会被跳过 —— "
            "可用 root 账号手工建号，或 `docker compose down -v && up -d` 重建；"
            "③ 改 .env 里的口令对**已存在**的账号无效，要用 root 执行 changeUserPassword。"
            "体检可跑 `python tools/mongo_check.py`")


def env_file_path() -> Path:
    """实际读取的连接参数文件：按顺序取**第一个存在的**，不是固定某一条路径。

    顺序 = `MONGO_ENV_FILE`（设了就在最前面）→ `mongo_configuration/.env` → `docker/.env`
    → 仓库根 `.env`；全都不存在时返回列表里的第一个（调用方会拿到空配置、走内置默认）。
    """
    chain = list(ENV_FILE_CANDIDATES)
    custom = (os.environ.get("MONGO_ENV_FILE") or "").strip()
    if custom:
        chain.insert(0, Path(custom).expanduser())
    for p in chain:
        if p.exists():
            return p
    return chain[0]


def __getattr__(name: str):
    """兼容旧的 `db.MONGO_ENV_FILE`：**每次访问都实时解析**，等价于 `db.env_file_path()`。

    以前它是导入时算好的 `ENV_FILE_CANDIDATES[0]`，前一个路径的文件不存在也不会往后找，
    所以 `.env` 稍后才创建（或 compose 目录改名）时，它一直是个错的（不存在的）值。
    """
    if name == "MONGO_ENV_FILE":
        return env_file_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _short_path(p: Path) -> str:
    """日志/接口里显示成相对仓库的短路径（不在仓库里就原样输出）。"""
    try:
        return str(p.relative_to(BASE_DIR))
    except ValueError:
        return str(p)


def read_env_file(path: Path | str | None = None) -> dict:
    """解析 `.env`（KEY=VALUE，支持 # 注释与引号）；不传路径就用 `env_file_path()`。"""
    out: dict[str, str] = {}
    p = Path(path) if path else env_file_path()
    if not p.exists():
        return out
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            out[key.strip()] = val.strip().strip('"').strip("'")
    except Exception:
        return {}
    return out


def _pick(names: tuple[str, ...], file_cfg: dict, keys: tuple[str, ...], default: str) -> str:
    for n in names:
        v = (os.environ.get(n) or "").strip()
        if v:
            return v
    for k in keys:
        v = (file_cfg.get(k) or "").strip()
        if v:
            return v
    return default


def _host_from_uri(uri: str) -> str:
    """从连接串里取出 `host:port`（去掉 scheme/凭据/库名/参数）。"""
    rest = uri.split("://", 1)[-1]
    rest = rest.rsplit("@", 1)[-1]
    return rest.split("/", 1)[0].split("?", 1)[0]


def resolve_settings() -> dict:
    """把环境变量 / `.env` / 默认值合成一份连接参数。"""
    envp = env_file_path()
    file_cfg = read_env_file(envp)
    host = _pick(("MONGO_HOST",), file_cfg, ("MONGO_HOST",), DEFAULTS["host"])
    port = _pick(("MONGO_PORT",), file_cfg, ("MONGO_PORT",), DEFAULTS["port"])
    user = _pick(("MONGO_USER",), file_cfg,
                 ("MONGO_APP_USER", "MONGO_ROOT_USER"), DEFAULTS["user"])
    password = _pick(("MONGO_PASSWORD",), file_cfg,
                     ("MONGO_APP_PASSWORD", "MONGO_ROOT_PASSWORD"), DEFAULTS["password"])
    auth_db = _pick(("MONGO_AUTH_DB",), file_cfg, ("MONGO_APP_DB",), DEFAULTS["auth_db"])
    dbname = _pick(("MONGO_DB",), file_cfg, ("MONGO_DB", "MONGO_APP_DB"), DEFAULTS["db"])

    uri = (os.environ.get("MONGO_URI") or file_cfg.get("MONGO_URI") or "").strip()
    env_used = any((os.environ.get(n) or "").strip() for n in (
        "MONGO_HOST", "MONGO_PORT", "MONGO_USER", "MONGO_PASSWORD",
        "MONGO_DB", "MONGO_AUTH_DB"))
    if uri:
        source = "MONGO_URI"
    elif env_used:
        source = "环境变量"          # 如 docker/ 里用 MONGO_HOST=mongo 连 compose 网络
    elif file_cfg:
        source = _short_path(envp)   # 显示实际读到的那份（候选里的哪一个）
    else:
        source = "默认值"
    if not uri:
        cred = ""
        if user:
            cred = f"{quote_plus(user)}:{quote_plus(password)}@" if password else f"{quote_plus(user)}@"
        auth = f"?authSource={quote_plus(auth_db)}" if user else ""
        uri = f"mongodb://{cred}{host}:{port}/{dbname}{auth}"
    else:
        host = _host_from_uri(uri)      # 以连接串为准，别显示成默认端口
        port = ""
    return {"uri": uri, "db": dbname, "host": host, "port": port, "user": user or "（无认证）",
            "auth_db": auth_db, "source": source, "env_file": str(envp)}


def mask_uri(uri: str) -> str:
    """`mongodb://user:pass@host` → `mongodb://user:****@host`（日志/接口用）。"""
    if "://" not in uri:
        return uri
    head, rest = uri.split("://", 1)
    if "@" not in rest:
        return uri
    cred, tail = rest.rsplit("@", 1)
    if ":" not in cred:
        return uri
    user, _ = cred.split(":", 1)
    return f"{head}://{user}:****@{tail}"


def preferred_backend() -> str:
    """APP_STORAGE = auto | mongo | json（默认 auto）。"""
    val = (os.environ.get("APP_STORAGE") or "").strip().lower()
    return val if val in ("auto", "mongo", "json") else "auto"


class Mongo:
    """MongoDB 客户端单例（懒连接 + 状态缓存）。"""

    def __init__(self, timeout_ms: int = DEFAULT_TIMEOUT_MS):
        self.cfg = resolve_settings()
        self.timeout_ms = timeout_ms
        self._lock = threading.RLock()
        self._client = None
        self._ok: bool | None = None
        self._error: str | None = None

    # ---------------- 连接 ----------------
    def _new_client(self):
        from pymongo import MongoClient
        return MongoClient(
            self.cfg["uri"],
            serverSelectionTimeoutMS=self.timeout_ms,
            connectTimeoutMS=self.timeout_ms,
            socketTimeoutMS=8000,
            tz_aware=False,
            appname="paper-trans-to-html",
        )

    def client(self):
        with self._lock:
            if self._client is None:
                self._client = self._new_client()
            return self._client

    def connect(self, force: bool = False) -> bool:
        """探测连通性；结果缓存，`force=True` 时重新探测。"""
        with self._lock:
            if self._ok is not None and not force:
                return self._ok
            try:
                self.client().admin.command("ping")
                self._ok, self._error = True, None
            except Exception as exc:
                self._ok = False
                self._error = short_error(exc)
                # 换掉旧 client，下次 refresh 拿到新的拓扑
                try:
                    if self._client is not None:
                        self._client.close()
                except Exception:
                    pass
                self._client = None
            return self._ok

    def refresh(self) -> bool:
        return self.connect(force=True)

    @property
    def available(self) -> bool:
        return self.connect()

    @property
    def error(self) -> str | None:
        self.connect()
        return self._error

    # ---------------- 库/集合 ----------------
    def db(self):
        if not self.available:
            raise StorageUnavailable(self._error or "MongoDB 不可用")
        return self.client()[self.cfg["db"]]

    def col(self, name: str):
        return self.db()[name]

    def status(self) -> dict:
        ok = self.available
        host = self.cfg["host"]
        if self.cfg.get("port"):
            host = f'{host}:{self.cfg["port"]}'
        return {
            "ok": ok,
            "db": self.cfg["db"],
            "host": host,
            "user": self.cfg["user"],
            "auth_db": self.cfg.get("auth_db"),
            "source": self.cfg.get("source"),
            "env_file": self.cfg.get("env_file"),
            "uri": mask_uri(self.cfg["uri"]),
            "error": self._error,
            # 认证失败时带上“怎么修”，前端/日志、/api/storage/status 都能直接用
            "hint": "" if ok else auth_hint(self._error, self.cfg),
        }


mongo = Mongo()
