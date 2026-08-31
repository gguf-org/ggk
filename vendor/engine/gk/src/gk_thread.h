#pragma once

// A minimal thread, mutex and condition-variable shim.
//
// Only what the pool below needs. pthreads everywhere except Windows, which
// gets the Win32 equivalents - the two APIs line up closely enough at this
// level that wrapping them is a few lines each and avoids a dependency on C11
// <threads.h>, which is still missing from MSVC.

#include <stdbool.h>
#include <stdint.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>

typedef HANDLE            gk_thread_t;
typedef CRITICAL_SECTION  gk_mutex_t;
typedef CONDITION_VARIABLE gk_cond_t;

static inline int gk_mutex_init(gk_mutex_t * m) {
    InitializeCriticalSection(m);
    return 0;
}
static inline void gk_mutex_destroy(gk_mutex_t * m) { DeleteCriticalSection(m); }
static inline void gk_mutex_lock   (gk_mutex_t * m) { EnterCriticalSection(m); }
static inline void gk_mutex_unlock (gk_mutex_t * m) { LeaveCriticalSection(m); }

static inline int gk_cond_init(gk_cond_t * c) {
    InitializeConditionVariable(c);
    return 0;
}
static inline void gk_cond_destroy  (gk_cond_t * c) { (void) c; }
static inline void gk_cond_wait     (gk_cond_t * c, gk_mutex_t * m) {
    SleepConditionVariableCS(c, m, INFINITE);
}
static inline void gk_cond_signal   (gk_cond_t * c) { WakeConditionVariable(c); }
static inline void gk_cond_broadcast(gk_cond_t * c) { WakeAllConditionVariable(c); }

typedef DWORD (WINAPI * gk_thread_fn)(LPVOID);

static inline int gk_thread_create(gk_thread_t * t, gk_thread_fn fn, void * arg) {
    *t = CreateThread(NULL, 0, fn, arg, 0, NULL);
    return *t == NULL ? -1 : 0;
}
static inline void gk_thread_join(gk_thread_t t) {
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
}

#  define GK_THREAD_RET     DWORD WINAPI
#  define GK_THREAD_RETURN  return 0

static inline int gk_cpu_count(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int) si.dwNumberOfProcessors;
}

#else // POSIX

#  include <pthread.h>
#  include <unistd.h>

typedef pthread_t       gk_thread_t;
typedef pthread_mutex_t gk_mutex_t;
typedef pthread_cond_t  gk_cond_t;

static inline int  gk_mutex_init   (gk_mutex_t * m) { return pthread_mutex_init(m, NULL); }
static inline void gk_mutex_destroy(gk_mutex_t * m) { pthread_mutex_destroy(m); }
static inline void gk_mutex_lock   (gk_mutex_t * m) { pthread_mutex_lock(m); }
static inline void gk_mutex_unlock (gk_mutex_t * m) { pthread_mutex_unlock(m); }

static inline int  gk_cond_init     (gk_cond_t * c) { return pthread_cond_init(c, NULL); }
static inline void gk_cond_destroy  (gk_cond_t * c) { pthread_cond_destroy(c); }
static inline void gk_cond_wait     (gk_cond_t * c, gk_mutex_t * m) { pthread_cond_wait(c, m); }
static inline void gk_cond_signal   (gk_cond_t * c) { pthread_cond_signal(c); }
static inline void gk_cond_broadcast(gk_cond_t * c) { pthread_cond_broadcast(c); }

typedef void * (*gk_thread_fn)(void *);

static inline int gk_thread_create(gk_thread_t * t, gk_thread_fn fn, void * arg) {
    return pthread_create(t, NULL, fn, arg);
}
static inline void gk_thread_join(gk_thread_t t) {
    pthread_join(t, NULL);
}

#  define GK_THREAD_RET    void *
#  define GK_THREAD_RETURN return NULL

static inline int gk_cpu_count(void) {
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n < 1 ? 1 : (int) n;
}

#endif
