"""MongoDB 连接体检（排查“连不上 / 悄悄回退本地 JSON”）

用法：
    python tools/mongo_check.py                  # 体检：打印生效参数 + 连通性 + 修复建议
    python tools/mongo_check.py --create-user    # 用 .env 里的 root 账号补建业务账号（可重复跑）

它回答三个问题：**参数从哪读的、实际拿哪个账号连、连不上时该改什么**。
口令一律打码，不会打印明文。

最常见的坑：MongoDB 的业务账号建在**业务库**里，而初始化脚本只在**数据卷为空**时
执行 —— 卷是旧的（或目录改名后 compose 没重建）就会被跳过，账号根本没建出来，
表现就是「能连到库，但 Authentication failed」，然后应用悄悄回退到本地 JSON。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.db import (ENV_FILE_CANDIDATES, auth_hint,  # noqa: E402
                        env_file_path, mask_uri, preferred_backend,
                        read_env_file, resolve_settings)


def _disp(p: Path) -> str:
    """显示成相对仓库的短路径（不在仓库里就原样输出）。"""
    try:
        return str(p.relative_to(BASE_DIR))
    except ValueError:
        return str(p)


def _hostport(cfg: dict) -> str:
    host = str(cfg.get("host") or "")
    if ":" in host or not cfg.get("port"):
        return host
    return f"{host}:{cfg['port']}"


def show_config() -> dict:
    cfg = resolve_settings()
    print("== 连接参数（实际生效）==")
    print(f"  uri         : {mask_uri(cfg['uri'])}")
    print(f"  业务库/认证库: {cfg['db']} / {cfg['auth_db']}")
    print(f"  账号        : {cfg['user']}")
    print(f"  地址        : {_hostport(cfg)}")
    print(f"  参数来源    : {cfg['source']}")
    print(f"  参数文件    : {cfg['env_file']}")
    chain = " → ".join(_disp(p) for p in ENV_FILE_CANDIDATES)
    print(f"               （按顺序取第一个存在的：{chain}；"
          f"可用 MONGO_ENV_FILE=路径 插到最前）")
    print(f"  APP_STORAGE : {preferred_backend()}（auto=连不上就回退 JSON / mongo=连不上直接报错）")
    return cfg


def check(cfg: dict) -> int:
    from backend.db import mongo

    print("\n== 连通性 ==")
    ok = mongo.refresh()
    if ok:
        print(f"  ✅ 已连接：{mask_uri(cfg['uri'])}")
        return 0
    print(f"  ❌ 连不上：{mongo.error}")
    hint = auth_hint(mongo.error, cfg)
    if hint:
        print(f"\n== 怎么修 ==\n  {hint}")
        print("\n  常用命令（把 <容器名> 换成 `docker ps` 里那个，通常是 paper-mongo）：")
        print(f"    # 看账号是否存在（要用 root 账号，按提示输密码）")
        print(f"    docker exec -it <容器名> mongosh -u <root用户> -p \\")
        print(f"      --authenticationDatabase admin "
              f"--eval 'db.getSiblingDB(\"{cfg['auth_db']}\").getUser(\"{cfg['user']}\")'")
        print(f"    # 没有就补建（口令取 .env 里的 MONGO_APP_PASSWORD）：")
        print(f"    python tools/mongo_check.py --create-user")
    else:
        print("  排查：容器是否在跑（docker ps）、端口是否映射到 "
              f"{_hostport(cfg)}、MONGO_HOST/PORT 是否写对。")
    return 1


def create_user(cfg: dict) -> int:
    """用 root 账号把业务账号补建出来（幂等；已存在则只提示）。"""
    try:
        from pymongo import MongoClient
    except ImportError:
        print("❌ 需要 pymongo：pip install -r requirements.txt")
        return 1

    env = read_env_file(env_file_path())
    root_user = (os.environ.get("MONGO_ROOT_USER") or env.get("MONGO_ROOT_USER") or "").strip()
    root_pwd = (os.environ.get("MONGO_ROOT_PASSWORD") or env.get("MONGO_ROOT_PASSWORD") or "").strip()
    app_user = (os.environ.get("MONGO_APP_USER") or env.get("MONGO_APP_USER") or cfg["user"]).strip()
    app_pwd = (os.environ.get("MONGO_APP_PASSWORD") or env.get("MONGO_APP_PASSWORD") or "").strip()
    target_db = cfg["auth_db"] or cfg["db"]

    if not (root_user and root_pwd):
        print("❌ 没找到 root 账号（需要在 .env 里配 MONGO_ROOT_USER / MONGO_ROOT_PASSWORD，"
              "或设成同名环境变量）")
        return 1
    if not (app_user and app_pwd):
        print("❌ 没找到业务账号口令（需要在 .env 里配 MONGO_APP_USER / MONGO_APP_PASSWORD）")
        return 1

    print(f"== 用 root 账号在 {target_db} 库里补建 {app_user} ==")
    uri = (f"mongodb://{quote_plus(root_user)}:{quote_plus(root_pwd)}@"
           f"{_hostport(cfg)}/?authSource=admin")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        db = client[target_db]
        try:
            found = [u for u in db.command("usersInfo", app_user).get("users", [])
                     if u.get("user") == app_user]
        except Exception:
            found = []
        if found:
            print(f"  ℹ️ {app_user}@{target_db} 已存在（改口令要用 changeUserPassword，"
                  f"或先 dropUser 再跑一次本命令）")
        else:
            db.command("createUser", app_user, pwd=app_pwd,
                       roles=[{"role": "readWrite", "db": cfg["db"]}])
            print(f"  ✅ 已创建 {app_user}@{target_db}（readWrite on {cfg['db']}）")
        client.close()
    except Exception as exc:
        print(f"  ❌ 失败：{type(exc).__name__}: {exc}")
        print("      （root 口令不对 / 容器没起都可能是这个原因）")
        return 1

    print("\n再跑一次体检确认：")
    return check(cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description="MongoDB 连接体检")
    ap.add_argument("--create-user", action="store_true",
                    help="用 .env 里的 root 账号补建业务账号（幂等）")
    args = ap.parse_args()

    cfg = show_config()
    if args.create_user:
        print()
        return create_user(cfg)
    return check(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
