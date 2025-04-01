import atexit
import signal
import sys
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait

from grpo_code.wasm import does_code_run, PythonWasmEnvironment

_executor = None


def worker_init(wasm_path: str, fuel: int):
    """
    Initialize a WASM environment for a worker process.

    Args:
        wasm_path (str): The path to the .wasm file.
        fuel (int): The amount of fuel available to the WASM environment.
    """
    import grpo_code.wasm as wasm

    wasm.worker_env = PythonWasmEnvironment(wasm_path, fuel)


def cleanup_executor():
    """
    Cleanup any running `ProcessPoolExecutor`.
    """
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


def cleanup_and_exit():
    """
    Gracefully shutsdown any running `ProcessPoolExecutor` on
    recieving a terminal signal.
    """
    cleanup_executor()
    sys.exit(0)


def get_multiprocessing_executor(max_processes: int, wasm_path: str, fuel: int):
    """
    Initialize a reusable `ProcessPoolExecutor` instance.

    Args:
        max_processes (int): The maximum number of processes to use.
        wasm_path (str): The path to the .wasm file.
        fuel (int): The amount of fuel available to the WASM environment.

    Returns:
        ProcessPoolExecutor: A `ProcessPoolExecutor` for parallel execution.
    """
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(
            max_workers=max_processes,
            initializer=worker_init,
            initargs=(wasm_path, fuel),
        )
        atexit.register(cleanup_executor)
        signal.signal(signal.SIGINT, cleanup_and_exit)
        signal.signal(signal.SIGTERM, cleanup_and_exit)
    return _executor


def run_tasks_with_multiprocessing_executor(
    executor: ProcessPoolExecutor, tasks: list[str], timeout: int
):
    """
    Run a list of code snippets in parallel by dispatching them to workers
    in a `ProcessPoolExecutor`.

    This function will gracefully handle timeout errors caused by workers, and
    recreate the `ProcessPoolExecutor` if necessary to avoid interupting training.

    Args:
        executor (ProcessPoolExecutor): The executor to use.
        tasks (list[str]): The list of code snippets to run.
        timeout (int): The timeout for each task.

    Returns:
        list[float]: The list of results from running the code snippets.
    """
    futures_to_index = {
        executor.submit(does_code_run, task): i for i, task in enumerate(tasks)
    }
    futures = list(futures_to_index)
    results = [0.0] * len(tasks)

    while futures:
        done, futures = wait(futures, timeout=timeout, return_when=FIRST_EXCEPTION)

        if futures and len(done) == 0:
            print(
                f"WARNING: Tasks timed out after {timeout} seconds, recreating process pool..."
            )
            cleanup_executor()
            break

        for future in done:
            task_index = futures_to_index[future]
            results[task_index] = future.result()

    return results
