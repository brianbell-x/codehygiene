static void stackInit(stack *ts) {
    /* Initialize the stack. */
    ts->items = ts->static_items;
    ts->count = 0;
}

static int applyRetries(job *jb) {
    // Step 1: validate the job
    int valid = validateJob(jb);
    // rest of the code remains unchanged
    return valid;
}
