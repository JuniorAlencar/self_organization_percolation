"""Bounded, ordered CPU parallelism shared by topology processors."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import argparse
import os


def default_jobs():
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return min(4, max(1, available - 1))


def positive_jobs(value):
    try:
        jobs = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("jobs must be a positive integer") from None
    if jobs < 1:
        raise argparse.ArgumentTypeError("jobs must be a positive integer")
    return jobs


def ordered_parallel_map(function, tasks, jobs):
    """Keep at most jobs tasks/results in flight, and preserve input order."""
    jobs = positive_jobs(jobs)
    iterator = iter(tasks)
    initial = []
    for _ in range(jobs):
        try:
            initial.append(next(iterator))
        except StopIteration:
            break
    if len(initial) <= 1:
        for task in initial:
            yield function(task)
        for task in iterator:
            yield function(task)
        return
    with ProcessPoolExecutor(max_workers=len(initial)) as executor:
        pending = deque(executor.submit(function, task) for task in initial)
        while pending:
            yield pending.popleft().result()
            try:
                task = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(function, task))
