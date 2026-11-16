/*
** How many times a failed write is retried after a locking error, and
** the delay before the first retry. The delay grows with each attempt.
*/
#ifndef APP_IOERR_RETRY
# define APP_IOERR_RETRY 10
#endif
#ifndef APP_IOERR_RETRY_DELAY
# define APP_IOERR_RETRY_DELAY 25
#endif
static int ioerrRetry = APP_IOERR_RETRY;
static int ioerrRetryDelay = APP_IOERR_RETRY_DELAY;
