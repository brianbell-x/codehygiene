fn recover_headers(journal: *Journal, chunk_buffer: []align(sector_size) const u8) void {
    // Store every redundant header in `journal.headers_redundant`, including the ones
    // that are invalid or corrupt. As the prepares are recovered, these headers are
    // replaced or removed as necessary.
    const chunk_headers = std.mem.bytesAsSlice(Header.Prepare, chunk_buffer);
    stdx.copy_disjoint(.exact, Header.Prepare, journal.headers_redundant, chunk_headers);
}
