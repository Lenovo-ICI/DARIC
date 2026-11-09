import sys
import argparse
import signal
import threading

from .local_store_service import LocalStoreManager
from ...logging import get_logger
from .zmq_rpc import ZMQRPCServer

logger = get_logger(__name__)


def start_local_store_service(storage_path, host, port, mem_max_size, disk_max_size):
    mem_max_size_bytes = int(mem_max_size * 1024 ** 3)
    disk_max_size_bytes = int(disk_max_size * 1024 ** 3)
    storage_mgr = LocalStoreManager(
        shm_size=mem_max_size_bytes,
        root_dir=storage_path,
        disk_size=disk_max_size_bytes,
    )

    targets = {"store": storage_mgr, "index": storage_mgr.index_manager}

    server = ZMQRPCServer(host=host, port=port, targets=targets, num_workers=16)
    server.start()
    logger.info(f"Local store server (ZMQ) started at {host}:{port}")

    shutdown_event = threading.Event()

    def signal_handler(sig, frame):
        logger.debug(f"Received signal {sig}, stopping server...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        shutdown_event.wait()
    except (KeyboardInterrupt, SystemExit):
        logger.debug("Local store server stopping...")
    finally:
        storage_mgr.close()
        server.stop()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start DARIC KV cache index/storage service."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50097,
        help="Port used by storage manager service.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host used by storage manager service.",
    )
    parser.add_argument(
        "--path",
        default=".kvcache",
        help="Disk storage path, only used when --type=disk.",
    )
    parser.add_argument(
        "--disk-size",
        type=float,
        default=64,
        help="Disk cache size in GB for disk/hybrid/local_store, or pinmem size in GB for pinmem.",
    )
    parser.add_argument(
        "--mem-size",
        type=float,
        default=16,
        help="Pinned memory cache size in GB, used by hybrid/local_store.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_local_store_service(
        storage_path=args.path,
        host=args.host,
        port=args.port,
        disk_max_size=args.disk_size,
        mem_max_size=args.mem_size,
    )


if __name__ == "__main__":
    main()