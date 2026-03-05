"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PROJECT MINECRAFT — Voxel Engine v1.0                         ║
║              Architecture: Senior Game Engine Developer                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Entry point with automatic dependency bootstrapper.

On first run (or whenever a package is missing), this script:
  1. Reads requirements.txt from its own directory
  2. Checks every non-commented, non-optional package via importlib
  3. Runs  pip install -r requirements.txt  for any that are missing
  4. Re-launches itself in a fresh process so new packages are on sys.path
  5. Starts the game (or headless server)
"""

# ── Bootstrap must use ONLY stdlib — nothing from requirements.txt yet ────────
import sys
import os
import subprocess
import importlib
import importlib.util
import re
import argparse

# ─── ANSI colour helpers (work on Windows 10+, macOS, Linux) ─────────────────
def _c(code: str, text: str) -> str:
    """Wrap text in an ANSI colour code if stdout is a real terminal."""
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

def _green(t):  return _c("92", t)
def _yellow(t): return _c("93", t)
def _red(t):    return _c("91", t)
def _cyan(t):   return _c("96", t)
def _bold(t):   return _c("1",  t)


# ─── Package name → importable module name mapping ───────────────────────────
# pip install name can differ from the Python import name.
IMPORT_NAMES = {
    "glfw":                 "glfw",
    "PyOpenGL":             "OpenGL",
    "PyOpenGL_accelerate":  "OpenGL_accelerate",
    "Pillow":               "PIL",
    "PyOpenAL":             "openal",
    "numpy":                "numpy",
    "pytest":               "pytest",
    "pytest-cov":           "pytest_cov",
}


# ─── Parse requirements.txt ───────────────────────────────────────────────────

def _find_requirements_file() -> str:
    """Locate requirements.txt relative to this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "requirements.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"requirements.txt not found at: {path}\n"
            "Make sure requirements.txt is in the same folder as main.py."
        )
    return path


def _parse_requirements(path: str) -> list:
    """
    Return a list of (pip_name, version_spec) tuples from requirements.txt.
    Skips blank lines, comments (#), and lines that start with '#' after
    stripping (handles inline comments too).
    Only returns lines that are NOT commented out with a leading '#'.
    """
    packages = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Skip blanks and full-line comments
            if not line or line.startswith("#"):
                continue

            # Strip inline comment
            if "#" in line:
                line = line[:line.index("#")].strip()

            if not line:
                continue

            # Skip options like --extra-index-url
            if line.startswith("-"):
                continue

            # Extract package name (everything before >=, <=, ==, !=, ~=, >)
            match = re.match(r"^([A-Za-z0-9_\-\.]+)", line)
            if match:
                pip_name = match.group(1)
                packages.append((pip_name, line))   # (name, full_spec)

    return packages


# ─── Check if a package is importable ────────────────────────────────────────

def _is_importable(pip_name: str) -> bool:
    """Return True if the package can be imported right now."""
    import_name = IMPORT_NAMES.get(pip_name, pip_name.replace("-", "_"))
    return importlib.util.find_spec(import_name) is not None


# ─── Main bootstrapper ───────────────────────────────────────────────────────

def bootstrap() -> bool:
    """
    Check all requirements. Install missing ones via pip.
    Returns True if everything was already installed (no action needed).
    Returns False if pip was run (caller should re-launch).
    Raises SystemExit on unrecoverable errors.
    """
    print(_bold(_cyan("\n╔══════════════════════════════════════════════╗")))
    print(_bold(_cyan(  "║   Project Minecraft — Dependency Checker     ║")))
    print(_bold(_cyan(  "╚══════════════════════════════════════════════╝\n")))

    try:
        req_path = _find_requirements_file()
    except FileNotFoundError as e:
        print(_red(f"[ERROR] {e}"))
        sys.exit(1)

    packages = _parse_requirements(req_path)
    if not packages:
        print(_yellow("[WARN] requirements.txt is empty or has no active packages."))
        return True

    print(f"  Checking {len(packages)} package(s) from requirements.txt ...\n")

    missing = []
    for pip_name, full_spec in packages:
        ok = _is_importable(pip_name)
        status = _green("  ✓  installed") if ok else _red("  ✗  MISSING  ")
        print(f"  {status}  {full_spec}")
        if not ok:
            missing.append(pip_name)

    print()

    if not missing:
        print(_green("  All requirements satisfied. Launching game...\n"))
        return True

    # ── Something is missing — run pip ───────────────────────────────────────
    print(_yellow(f"  {len(missing)} package(s) need to be installed:"))
    for m in missing:
        print(_yellow(f"    • {m}"))
    print()
    print(_bold("  Running:  pip install -r requirements.txt\n"))
    print("─" * 50)

    pip_cmd = [sys.executable, "-m", "pip", "install", "-r", req_path]

    try:
        result = subprocess.run(pip_cmd, check=False)
    except Exception as e:
        print(_red(f"\n[ERROR] Failed to run pip: {e}"))
        print(_yellow("  Try running manually:\n    pip install -r requirements.txt"))
        sys.exit(1)

    print("─" * 50)

    if result.returncode != 0:
        print(_red("\n[ERROR] pip exited with errors (see above)."))
        print(_yellow("  Some packages may need system-level libraries first:"))
        print(_yellow("    Linux:   sudo apt install libglfw3-dev libopenal-dev libgl1-mesa-dev"))
        print(_yellow("    macOS:   brew install glfw openal-soft"))
        print(_yellow("    Windows: see requirements.txt PLATFORM NOTES"))
        sys.exit(result.returncode)

    print(_green("\n  All packages installed successfully!"))
    return False   # Signal: re-launch needed so new packages load cleanly


# ─── Re-launcher ─────────────────────────────────────────────────────────────

def relaunch():
    """
    Replace the current process with a fresh Python process running this
    same script with the same arguments. This ensures newly-installed
    packages appear on sys.path without any import-cache issues.
    """
    print(_cyan("  Re-launching with fresh Python environment...\n"))
    # Pass __VOXEL_RELAUNCHED__ env var so we don't loop infinitely
    env = os.environ.copy()
    env["__VOXEL_RELAUNCHED__"] = "1"
    cmd = [sys.executable] + sys.argv
    os.execve(sys.executable, cmd, env)   # replaces current process (Unix)
    # os.execve doesn't return. On Windows it falls through:
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


# ─── Game launcher ───────────────────────────────────────────────────────────

def launch_game():
    """Parse CLI args and start the game or headless server."""
    parser = argparse.ArgumentParser(
        description="Project Minecraft — Voxel Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        Start the game (default seed)
  python main.py --seed 99999           Custom world seed
  python main.py --headless             Run as dedicated server
  python main.py --headless --port 7777 Server on custom port
        """
    )
    parser.add_argument("--headless", action="store_true",
                        help="Run as a headless dedicated server (no window)")
    parser.add_argument("--host",  default="0.0.0.0",
                        help="Server bind address (headless only, default: 0.0.0.0)")
    parser.add_argument("--port",  type=int, default=25565,
                        help="Server port (headless only, default: 25565)")
    parser.add_argument("--seed",  type=int, default=12345,
                        help="World generation seed (default: 12345)")
    parser.add_argument("--world", default="world",
                        help="World save name / directory (default: world)")
    parser.add_argument("--skip-check", action="store_true",
                        help="Skip dependency check and launch immediately")
    args = parser.parse_args()

    if args.headless:
        from engine.server import DedicatedServer
        server = DedicatedServer(host=args.host, port=args.port,
                                  seed=args.seed, world_name=args.world)
        server.run()
    else:
        from engine.game import Game
        game = Game(seed=args.seed, world_name=args.world)
        game.run()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Detect if we've already been re-launched to prevent infinite loops
    already_relaunched = os.environ.get("__VOXEL_RELAUNCHED__") == "1"

    # Allow bypassing the check entirely with --skip-check
    skip_check = "--skip-check" in sys.argv

    if skip_check or already_relaunched:
        # Straight to game — no dependency check
        if already_relaunched:
            print(_green("  [OK] Dependencies verified. Starting game...\n"))
        launch_game()
    else:
        # Run the bootstrapper
        all_good = bootstrap()

        if all_good:
            # Everything was already installed — launch directly
            launch_game()
        else:
            # pip just ran — re-launch so new packages are importable
            relaunch()
