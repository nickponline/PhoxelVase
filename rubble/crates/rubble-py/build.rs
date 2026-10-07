fn main() {
    // macOS: allow unresolved libpython symbols so plain `cargo build --workspace` links.
    pyo3_build_config::add_extension_module_link_args();
}
