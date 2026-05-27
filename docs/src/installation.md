# Installation

FeRx requires a nightly Rust toolchain with the Enzyme LLVM plugin for automatic differentiation. As of 2026, Enzyme is not yet distributed via rustup, so a one-time plugin build is required.

Pick your platform:

- [**Linux**](#linux) — fully supported, one-time ~30 min plugin build
- [**macOS**](#macos) — supported with caveats
- [**Windows**](#windows) — **not supported** (see [why below](#windows))

---

## Linux

Tested on Ubuntu 22.04. Other distributions (Debian, Fedora, Arch) should work with adjustments to the package manager commands.

### 1. Install rustup + upstream nightly

**Do not use snap's rustup** — its filesystem confinement breaks on non-standard home directories (common on enterprise servers).

```bash
# Remove snap rustup if you had it:
sudo snap remove rustup

# Official installer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

rustup toolchain install nightly
```

### 2. Install system dependencies and matching LLVM

Check which LLVM major version nightly needs:
```bash
rustc +nightly --version --verbose | grep LLVM
# e.g. "LLVM version: 22.1.2" — major is 22
```

Use that major version (`<MAJOR>`) below.

```bash
sudo apt install -y cmake ninja-build clang libssl-dev pkg-config \
                    python3 build-essential curl git libzstd-dev

# Install matching LLVM from apt.llvm.org (Ubuntu's defaults lag behind):
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh <MAJOR>

# Fix GPG keyring permissions if apt warns:
sudo chmod 644 /etc/apt/trusted.gpg.d/apt.llvm.org.asc
sudo apt update

sudo apt install -y llvm-<MAJOR>-dev clang-<MAJOR>
```

For other distros, install the matching llvm-dev + clang packages from your package manager. The LLVM major version must exactly match what rustc reports.

### 3. Build the Enzyme plugin

```bash
git clone https://github.com/EnzymeAD/Enzyme /tmp/enzyme-build
cd /tmp/enzyme-build/enzyme
mkdir build && cd build

cmake -G Ninja .. \
  -DLLVM_DIR=/usr/lib/llvm-<MAJOR>/lib/cmake/llvm \
  -DENZYME_CLANG=OFF \
  -DENZYME_FLANG=OFF
ninja
# A few minutes when building against the prebuilt llvm-<MAJOR>-dev package
# installed above; 15–30 min only if you had to build LLVM itself from source.
```

### 4. Install the plugin into nightly's sysroot

**This location is not obvious** — rustc looks in `lib/rustlib/<target>/lib/`, not just `lib/`. Despite the error wording ("folder not found"), it's searching for a file:

```bash
SYSROOT=$(rustc +nightly --print sysroot)
TARGET=x86_64-unknown-linux-gnu   # or aarch64-unknown-linux-gnu on ARM

cp /tmp/enzyme-build/enzyme/build/Enzyme/LLVMEnzyme-<MAJOR>.so \
   $SYSROOT/lib/rustlib/$TARGET/lib/libEnzyme-<MAJOR>.so
```

Note the filename rewrite: `LLVMEnzyme-<N>.so` → `libEnzyme-<N>.so` (with `lib` prefix).

### 5. Register the toolchain as `enzyme`

ferx's build system pins to a toolchain named `enzyme`:

```bash
rustup toolchain link enzyme "$(rustc +nightly --print sysroot)"
rustc +enzyme --version
```

### 6. Verify

```bash
rustc +enzyme -Z autodiff=Enable - </dev/null 2>&1 | head
```

Expected: `error[E0601]: `main` function not found`. That's the success signal. If you see `autodiff backend not found in the sysroot`, the `.so` is missing or in the wrong place (step 4).

### Multi-user servers

For shared Linux servers, stage the built toolchain in `/opt/rust-nightly` so all users can share one build:

```bash
sudo mkdir -p /opt/rust-nightly
sudo cp -a ~/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/. /opt/rust-nightly/
sudo chown -R root:root /opt/rust-nightly
sudo chmod -R a+rX /opt/rust-nightly
sudo chmod a+rx /opt/rust-nightly/bin/*
```

Each user then links the shared toolchain into their own rustup:
```bash
# per user, once
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
rustup toolchain link enzyme /opt/rust-nightly
```

For R users, add to `~/.Renviron`:
```
PATH=/opt/rust-nightly/bin:${HOME}/.cargo/bin:${PATH}
RUSTUP_TOOLCHAIN=enzyme
```

---

## macOS

Supported on both Intel and Apple Silicon, with caveats around LLVM installation.

### 1. Install Xcode Command Line Tools

```bash
xcode-select --install
```

### 2. Install rustup + upstream nightly

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

rustup toolchain install nightly
```

### 3. Install matching LLVM via Homebrew

Check which LLVM major version nightly needs:
```bash
rustc +nightly --version --verbose | grep LLVM
```

Homebrew only ships a handful of LLVM versions at a time.

**If the major you need is Homebrew's current `llvm`**, there is no `llvm@<MAJOR>` versioned formula — install plain `llvm` instead. (Check with `brew info llvm`; e.g. in mid-2026 `brew install llvm` gives 22.1.6, which is the LLVM 22 a current nightly wants.) In that case the path is `$(brew --prefix)/opt/llvm/lib/cmake/llvm` — no version suffix — for the `LLVM_DIR` below.

**If you need an older major** that Homebrew still keeps as a versioned formula:
```bash
brew install llvm@<MAJOR>
# Path will be /opt/homebrew/opt/llvm@<MAJOR> (Apple Silicon)
# or           /usr/local/opt/llvm@<MAJOR>     (Intel)
```

`llvm` is keg-only (not symlinked into your `PATH`), which is fine — we point `LLVM_DIR` at it explicitly.

If Homebrew has neither, you'll need to build LLVM from source (much longer — consider sticking with a slightly older nightly that matches an available LLVM version).

### 4. Build the Enzyme plugin

The Enzyme repo's CMake project lives in the `enzyme/` **subdirectory** of the clone, and that directory already contains a Bazel `BUILD` file. On macOS's case-insensitive filesystem `mkdir build` collides with `BUILD`, so use a different out-of-source build directory name (e.g. `cmake-build`) and pass the source path explicitly:

```bash
git clone https://github.com/EnzymeAD/Enzyme /tmp/enzyme-build
cd /tmp/enzyme-build/enzyme
mkdir cmake-build && cd cmake-build

# For plain `llvm`, drop the @<MAJOR> suffix from LLVM_DIR (see step 3).
# Adjust the brew prefix as needed (/opt/homebrew on Apple Silicon, /usr/local on Intel).
cmake -G Ninja /tmp/enzyme-build/enzyme \
  -DLLVM_DIR=$(brew --prefix)/opt/llvm@<MAJOR>/lib/cmake/llvm \
  -DENZYME_CLANG=OFF \
  -DENZYME_FLANG=OFF
ninja
```

You may need to install `ninja` if not present: `brew install ninja cmake`.

**Build time depends entirely on whether you're reusing a prebuilt LLVM.** When `LLVM_DIR` points at a precompiled LLVM (a Homebrew bottle, a distro `llvm-<MAJOR>-dev` package, etc.), `ninja` only compiles Enzyme itself — typically **under a minute to a few minutes** on Apple Silicon, so a sub-minute build is expected and not a sign of a partial build. The 15–30 min figure quoted elsewhere applies only when LLVM itself has to be built from source (the case when Homebrew/your package manager doesn't carry the major version you need). To confirm a fast build is complete rather than truncated, check the plugin is a full-size shared library (tens of MB), e.g. `ls -lh .../Enzyme/LLVMEnzyme-<MAJOR>.dylib`.

### 5. Install the plugin into nightly's sysroot

```bash
SYSROOT=$(rustc +nightly --print sysroot)

# On Apple Silicon:
TARGET=aarch64-apple-darwin
# On Intel:
# TARGET=x86_64-apple-darwin

# Note: on macOS the shared library extension is .dylib, not .so
cp /tmp/enzyme-build/enzyme/cmake-build/Enzyme/LLVMEnzyme-<MAJOR>.dylib \
   $SYSROOT/lib/rustlib/$TARGET/lib/libEnzyme-<MAJOR>.dylib
```

(If the exact filename differs, `ls /tmp/enzyme-build/enzyme/cmake-build/Enzyme/` to confirm the `LLVMEnzyme-<MAJOR>.dylib` name.)

### 6. Register and verify

```bash
rustup toolchain link enzyme "$(rustc +nightly --print sysroot)"
rustc +enzyme -Z autodiff=Enable - </dev/null 2>&1 | head
# Expect: error[E0601]: `main` function not found
```

### macOS caveats

- **Apple Silicon (M1/M2/M3/M4)**: use `aarch64-apple-darwin` as `TARGET`. Intel Macs use `x86_64-apple-darwin`
- **Case-insensitive filesystem**: the default macOS filesystem treats `build` and the Enzyme source tree's `BUILD` (Bazel) file as the same name, so `mkdir build` fails with `File exists`. Use a different build-directory name such as `cmake-build` (as shown above)
- **System Integrity Protection (SIP)**: If you see code-signing errors loading the `.dylib` during `rustc` invocation, try `sudo codesign --force --sign - <path-to-libEnzyme.dylib>` — should be rare on dev machines
- **Nightly toolchain distribution mismatches**: Apple Silicon nightlies occasionally lag x86_64 by a day or two; if LLVM version mismatches after `rustup update`, prefer installing a specific dated nightly

---

## Windows

**Windows is not currently supported.** The blockers are:

1. **EnzymeAD plugin build on MSVC is not well-tested.** The Enzyme project primarily targets Linux with LLVM clang/gcc. While Windows builds exist in theory, the toolchain integration (rustc sysroot path conventions, plugin discovery, MSVC vs MinGW linking) has known issues that haven't been worked through.

2. **Rust autodiff feature gate interactions with MSVC codegen.** The `-Z autodiff=Enable` path has not been exercised on the x86_64-pc-windows-msvc target. Anecdotal reports from the Rust autodiff tracking issue show crashes or "backend not found" errors on Windows even when the plugin `.dll` is present.

3. **We haven't set up CI for Windows.** Without a green CI signal we can't promise the build works.

### Workarounds for Windows users

- **WSL2 (recommended)**: Install Ubuntu under WSL2 and follow the [Linux instructions](#linux). Performance is near-native for CPU-bound workloads like model fitting.
- **Docker**: Use the forthcoming ferx-core Docker image (coming soon) which ships with the toolchain pre-installed.
- **Remote dev**: SSH into a Linux server or cloud VM.

### Future Windows support

Once upstream Enzyme integration lands in rustup (tracked at [rust-lang/rust autodiff tracking issue](https://github.com/rust-lang/rust/issues/124509)), Windows support should become straightforward since the plugin distribution problem goes away. No ETA; the feature is still experimental in rustc.

If you're a Windows developer who would like to help, a CI run against `x86_64-pc-windows-msvc` + contributed docs for that platform would be very welcome.

---

## Building ferx-core from source

Once your Enzyme toolchain is set up:

```bash
git clone https://github.com/FeRx-NLME/ferx-core
cd ferx-core

RUSTFLAGS="-Z autodiff=Enable" cargo build --release --features autodiff
# Binary at target/release/ferx
```

> **`RUSTFLAGS="-Z autodiff=Enable"` is required for every `--features autodiff` build.**
> The `autodiff` feature won't compile without it — you'll get
> `error: using the autodiff feature requires -Z autodiff=Enable`.
>
> Pass it inline as shown (it's the form CI and the R package use), or export it for
> the current shell with `export RUSTFLAGS="-Z autodiff=Enable"`.
>
> Do **not** bake it into a committed `.cargo/config.toml`: that flag is nightly-only
> and applies to *all* builds, which breaks the stable toolchain and the no-Enzyme
> `--features ci` path (including CI). If you want it persistent on your own machine,
> keep it in a *git-ignored* local `.cargo/config.toml` or a shell/`direnv` export.

### Build options

```bash
# Debug build (faster compile, slower runtime)
RUSTFLAGS="-Z autodiff=Enable" cargo build --features autodiff

# Quick type-check without building
RUSTFLAGS="-Z autodiff=Enable" cargo check --features autodiff

# Lints
RUSTFLAGS="-Z autodiff=Enable" cargo clippy --features autodiff

# CI build without autodiff (uses finite differences — no Enzyme, no RUSTFLAGS needed)
cargo build --release --no-default-features --features ci
```

The `ci` feature is useful for development on machines without the full Enzyme toolchain — at the cost of much slower gradient computation. It builds on plain nightly (no Enzyme plugin) and does not need the `RUSTFLAGS` flag.

### Verify the build

```bash
RUSTFLAGS="-Z autodiff=Enable" cargo run --release --features autodiff --bin ferx -- examples/warfarin.ferx --simulate
```

Should print a successful model fit with parameter estimates.

---

## Installing the ferx R package

The R package handles all of the above automatically — you do **not** set `RUSTFLAGS` yourself. Just:

```r
devtools::install_github("FeRx-NLME/ferx-r")
```

The package's build is driven by its `src/Makevars`, which:

1. **Probes for the `enzyme` toolchain** with `rustup toolchain list | grep -q '^enzyme'`.
2. **If Enzyme is present**, it builds the autodiff path: it writes `rustflags = ["-Z", "autodiff=Enable"]` into a generated `rust/.cargo/config.toml` and pins `channel = "enzyme"` in a generated `rust/rust-toolchain.toml` (both build artifacts, not committed), so the flag from option #1 above is supplied for you.
3. **If Enzyme is missing**, it silently falls back to the stable path (`--no-default-features --features ci,nn`, thin LTO) so the install still succeeds — with finite-difference gradients instead of autodiff. The package prints a notice on load when built this way.

You can override the probe with an environment variable before installing:

- `FERX_NO_AUTODIFF=1` — force the stable (no-autodiff) build even if Enzyme is installed.
- `FERX_NO_AUTODIFF=0` — force the autodiff build, erroring out if the `enzyme` toolchain isn't found.

To confirm which path your installed package took, call `ferx:::ferx_rust_autodiff_enabled()` in R.

This is exactly why option #2 (a committed `.cargo/config.toml`) is discouraged for the core repo: the R package's `Makevars` already manages the flag conditionally per machine, generating the config only when Enzyme is actually available. A committed flag in `ferx-core` would conflict with that logic and break the no-Enzyme install.

See the [ferx R package README](https://github.com/FeRx-NLME/ferx-r) for API usage.

### Cancelling a running fit (Ctrl-C)

`ferx_fit()` runs the estimator on a worker thread and polls for R interrupts on the main thread every ~100 ms, so **Ctrl-C** (or RStudio's red stop button) aborts the fit cleanly. The worker exits at the next safe checkpoint — typically within a second or two, but up to one inner-loop evaluation for heavy ODE models. The call then returns with an R error:

```
Error: ferx_fit: cancelled by user
```

Ctrl-Z (SIGTSTP) will *not* abort the fit — it suspends the whole R process to the shell. Use Ctrl-C instead.

---

## Dependencies

FeRx depends on these crates (managed automatically by Cargo):

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Linear algebra (matrices, Cholesky) |
| `nlopt` | Nonlinear optimization (SLSQP, L-BFGS, MMA) |
| `rayon` | Parallel computation |
| `rand`, `rand_distr` | Random number generation (SAEM, SIR) |
| `csv` | CSV data file reading |
| `regex` | Model file expression parsing |

The `nlopt` crate requires the NLopt C library. Most platforms handle this automatically; if the build fails on NLopt, install it via your system package manager:
```bash
# macOS
brew install nlopt

# Ubuntu/Debian
sudo apt install libnlopt-dev

# Fedora
sudo dnf install NLopt-devel
```

---

## Troubleshooting

### `"error: the option 'Z' is only accepted on the nightly compiler"`
Your shell (or R) is finding a non-nightly `rustc`. Check `rustc --version` and your `PATH`. For R, verify `Sys.which("rustc")` and `~/.Renviron`.

### `"autodiff backend not found in the sysroot: failed to find a libEnzyme-<N> folder"`
Despite the wording ("folder"), rustc is searching for a file. Causes:
- **Wrong directory**: the `.so`/`.dylib` is in `<sysroot>/lib/` instead of `<sysroot>/lib/rustlib/<target>/lib/`
- **LLVM version mismatch**: rebuild Enzyme against the LLVM version rustc reports
- **Filename**: must be `libEnzyme-<MAJOR>.so` (Linux) or `libEnzyme-<MAJOR>.dylib` (macOS), with `lib` prefix

### `"custom toolchain 'enzyme' specified in override file ... is not installed"`
Run `rustup toolchain link enzyme <path-to-nightly-sysroot>`.

### `"not a directory: '/<path>/lib'"` from `rustup toolchain link`
Permission issue — the user running `toolchain link` can't read the target path. On shared installs, run `sudo chmod -R a+rX /opt/rust-nightly`.

### `"incorrect value 'X' for unstable option 'autodiff'"`
Valid autodiff values change between nightly builds. Test with:
```bash
rustc +enzyme -Z autodiff=Enable - </dev/null 2>&1 | head
```
If `Enable` is rejected, try `LooseTypes`, `Inline`, or `PrintTA`.

### `"Enzyme: cannot handle (forward) unknown intrinsic llvm.maximumnum"`
Recent rustc lowers `f64::max()`/`f64::min()` to intrinsics Enzyme can't yet differentiate. This is a ferx-core code-level issue — AD-instrumented functions must use manual `if` expressions. Should not happen on released ferx-core versions; if it does, please file an issue.

### `"cargo is unavailable for the active toolchain"` (info, not error)
Cargo wasn't copied into your linked toolchain. Either copy it (`cp ~/.cargo/bin/cargo /opt/rust-nightly/bin/`) or ignore — rustup falls back to nightly's cargo, which works.

### Refreshing when upstream nightly rolls forward
If a new ferx-core release references stdlib items your cached toolchain doesn't have (e.g. `autodiff_forward` not found), rebuild `/opt/rust-nightly` against the current nightly, and rebuild Enzyme if the LLVM major version changed.
