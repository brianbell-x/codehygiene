typedef struct cacheType {
    unsigned long (*hashFunction)(const void *key);
    /* Lets a cache carry extra caller-defined metadata. The
     * extra memory is initialized to 0 when the cache is allocated. */
    size_t (*metadataBytes)(cache *c);
    /* Called once the size of the cache changes. */
    void (*sizeChanged)(cache *c, long long delta);
} cacheType;
