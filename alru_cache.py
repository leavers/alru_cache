import weakref
from functools import _CacheInfo, _make_key, partial, partialmethod, update_wrapper
from threading import RLock
from typing import Any, Callable, ParamSpec, TypeVar, cast

__version__ = "0.0.1"

P = ParamSpec("P")
T = TypeVar("T")


def alru_cache(
    func: Callable | partial | partialmethod | None = None,
    maxsize: int | None = 128,
    typed: bool = False,
    *,
    weak_self: bool = True,
):
    """Customized LRU cache for asynchoronized functions."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        org_func = func
        while isinstance(org_func, (partial, partialmethod)):
            org_func = org_func.func
        varnames = org_func.__code__.co_varnames
        if (
            weak_self  # enable logic for `self`
            # # As the class of func is under construction during invoking lru_cache,
            # # getmethodclass(func) would not work here, and isfunction is always True
            # and getmethodclass(func) is not None
            # and not isfunction(func)
            and varnames
            and varnames[0] == "self"
            and not hasattr(func, "__self__")  # not a classmethod
        ):
            if hasattr(func, "_make_unbound_method"):
                func = func._make_unbound_method()

            def _weak_self(self_ref, *args, **kwargs):
                return func(self_ref(), *args, **kwargs)

            _weak_wrapper = _lru_cache_wrapper(_weak_self, maxsize, typed, _CacheInfo)

            def wrapper(self, *args, **kwargs):
                return _weak_wrapper(weakref.ref(self), *args, **kwargs)

            wrapper = update_wrapper(wrapper, func)
            setattr(wrapper, "__weak_self__", True)
            return cast(Callable[P, T], wrapper)
        else:
            wrapper = _lru_cache_wrapper(func, maxsize, typed, _CacheInfo)
            return cast(Callable[P, T], update_wrapper(wrapper, func))

    if func is not None:
        return decorator(func)  # type: ignore[arg-type]

    return decorator


def _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo):
    # Constants shared by all lru cache instances:
    sentinel = object()  # unique object used to signal cache misses
    make_key = _make_key  # build a key from the function arguments
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3  # names for the link fields

    cache: dict[Any, Any] = {}
    hits = misses = 0
    full = False
    cache_get = cache.get  # bound method to lookup a key or return None
    cache_len = cache.__len__  # get cache size without calling len()
    lock = RLock()  # linkedlist updates aren't threadsafe
    root: list[Any] = []  # root of the circular doubly linked list
    root[:] = [root, root, None, None]  # initialize by pointing to self

    if maxsize == 0:

        async def wrapper(*args, **kwds):
            # No caching -- just a statistics update
            nonlocal misses
            misses += 1
            result = await user_function(*args, **kwds)
            return result

    elif maxsize is None:

        async def wrapper(*args, **kwds):
            # Simple caching without ordering or size limit
            nonlocal hits, misses
            key = make_key(args, kwds, typed)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                hits += 1
                return result
            misses += 1
            result = await user_function(*args, **kwds)
            cache[key] = result
            return result

    else:

        async def wrapper(*args, **kwds):
            # Size limited caching that tracks accesses by recency
            nonlocal root, hits, misses, full
            key = make_key(args, kwds, typed)
            with lock:
                link = cache_get(key)
                if link is not None:
                    # Move the link to the front of the circular queue
                    link_prev, link_next, _key, result = link
                    link_prev[NEXT] = link_next
                    link_next[PREV] = link_prev
                    last = root[PREV]
                    last[NEXT] = root[PREV] = link
                    link[PREV] = last
                    link[NEXT] = root
                    hits += 1
                    return result
                misses += 1
            result = await user_function(*args, **kwds)
            with lock:
                if key in cache:
                    # Getting here means that this same key was added to the
                    # cache while the lock was released.  Since the link
                    # update is already done, we need only return the
                    # computed result and update the count of misses.
                    pass
                elif full:
                    # Use the old root to store the new key and result.
                    oldroot = root
                    oldroot[KEY] = key
                    oldroot[RESULT] = result
                    # Empty the oldest link and make it the new root.
                    # Keep a reference to the old key and old result to
                    # prevent their ref counts from going to zero during the
                    # update. That will prevent potentially arbitrary object
                    # clean-up code (i.e. __del__) from running while we're
                    # still adjusting the links.
                    root = oldroot[NEXT]
                    oldkey = root[KEY]
                    root[KEY] = root[RESULT] = None
                    # Now update the cache dictionary.
                    del cache[oldkey]
                    # Save the potentially reentrant cache[key] assignment
                    # for last, after the root and links have been put in
                    # a consistent state.
                    cache[key] = oldroot
                else:
                    # Put result in a new link at the front of the queue.
                    last = root[PREV]
                    link = [last, root, key, result]
                    last[NEXT] = root[PREV] = cache[key] = link
                    # Use the cache_len bound method instead of the len() function
                    # which could potentially be wrapped in an lru_cache itself.
                    full = cache_len() >= maxsize
            return result

    def cache_info():
        """Report cache statistics"""
        with lock:
            return _CacheInfo(hits, misses, maxsize, cache_len())

    def cache_clear():
        """Clear the cache and cache statistics"""
        nonlocal hits, misses, full
        with lock:
            cache.clear()
            root[:] = [root, root, None, None]
            hits = misses = 0
            full = False

    wrapper.cache_info = cache_info  # type: ignore[attr-defined]
    wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
    return wrapper
