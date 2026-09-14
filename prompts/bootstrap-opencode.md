# Bootstrap prompt for friend's MacBook — paste into OpenCode after running setup-friend-mac.sh
# File: prompts/bootstrap-opencode.md
# Usage on MacBook (after `opencode` is installed):
#   cd ~/Desktop/drone-nav-sar
#   opencode run "$(cat prompts/bootstrap-opencode.md)" -m muse-spark-1.3-contributor-free
# Or just open opencode TUI and paste the text below.

You are bootstrapping the DroneNav-SAR project on a NEW MacBook (friend's machine).

Repo: https://github.com/pandejesal/drone-nav-sar (private)
Vault submodule: https://github.com/pandejesal/obsidian-vault (private, at ./vault)

Do ALL of the following, in order, and report what you did:

1. VERIFY REPO INTEGRITY
   - `git status` should be clean (master tracking origin/master)
   - `git submodule status` should show vault initialized (or clone fallback at ./vault)
   - If .gitmodules missing, warn — owner must push it from Windows machine.

2. VERIFY ISOLATION (critical — do NOT break owner's other projects)
   - Confirm project-local paths only:
     - .venv at $PWD/.venv (not global python)
     - .opencode/opencode.json (project-local, free-tier rotation)
     - .harness-memory/evolution-lessons.json (project-local, tracked)
     - vault at $PWD/vault (submodule, isolated — global ~/Documents/Obsidian Vault untouched)
   - Do NOT write to ~/Documents/Obsidian Vault, ~/.config/hermes global config, or other repos on disk.
   - If you find global configs, leave them alone.

3. INSTALL & VERIFY DEPS INSIDE .venv
   - Activate .venv: source .venv/bin/activate
   - Ensure installed: gymnasium, numpy, pytest, torch (CPU ok on Mac), pyyaml
   - If requirements.txt / pyproject.toml exists, install from it; else pip install the list above.

4. RUN TESTS (headless, no GPU needed)
   - `python -m pytest tests/test_reconstruction.py tests/test_sim.py -q`
   - All sim tests should use mock_backend (no Isaac/AirSim/Gazebo required on Mac).
   - Report pass/fail. If failures, do NOT auto-fix beyond missing deps — record in 04-Prompt-Queues/Ecosystem/InterHarness/mac-bootstrap-YYYY-MM-DD.md

5. WRITE BOOTSTRAP REPORT
   - Create 04-Prompt-Queues/Ecosystem/InterHarness/mac-bootstrap-$(date +%F).md with:
     - Mac model, macOS version, python version, node version
     - git status + submodule status
     - pytest output summary
     - Isolation check result (confirming global vault/hermes untouched)

6. NEXT STEPS FOR HUMAN
   - If tests green: you are ready for Sprint 7 (SAR scenario). Read prompt.md § OWNERSHIP MAP — your lane is Sprints 7-9.
   - If auth errors on vault (private repo): run `gh auth login` then `git submodule update --init --recursive`.

Constraints:
- $0 runtime, free-tier models only (muse-spark-1.3-contributor-free, nemotron-3-ultra-free).
- Do not push to origin — bootstrap report stays local until human reviews.
- Do not delete or rewrite owner's prompt.md ownership map — append your report to InterHarness bus only.
