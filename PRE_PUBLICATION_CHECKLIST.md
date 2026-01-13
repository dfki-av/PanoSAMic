# PanoSAMic Pre-Publication Checklist

This checklist identifies items that should be addressed before making the repository public.

## **CRITICAL ISSUES** (Must Fix)

- [x] **License - Line 11 in pyproject.toml**
  - ~~Currently set to `license = { text = "INTERNAL" }`~~
  - ~~Specify a proper open-source license (e.g., MIT, Apache 2.0, GPL-3.0) or proprietary license terms~~
  - ~~Without a proper license, others cannot legally use your code~~
  - **COMPLETED**: Set to CC BY-NC-SA 4.0

- [x] **Missing LICENSE File**
  - ~~No LICENSE file exists in the repository root~~
  - ~~Required for GitHub and proper licensing~~
  - **COMPLETED**: LICENSE file added with CC BY-NC-SA 4.0 text

- [x] **Missing Key Documentation Files**
  - [x] ~~No CITATION.cff or CITATION.bib (if this is research code, provide citation information)~~
    - **DECISION**: Citation will be added to README when paper is published, no separate .cff/.bib file needed
  - [x] ~~No CONTRIBUTING.md (guidelines for how others can contribute)~~
    - **DECISION**: Not needed - this is a research project related to a paper, not accepting contributions
  - [x] ~~No requirements.txt (while you have pyproject.toml, a requirements.txt is often expected)~~
    - **DECISION**: pyproject.toml with uv is sufficient for this project

## **HIGH PRIORITY** (Strongly Recommended)

- [x] **README Improvements**
  - [x] ~~Add project description/abstract explaining what PanoSAMic does~~
    - **COMPLETED**: Added comprehensive project description based on paper abstract
  - [x] ~~Add usage examples or quickstart guide~~
    - **COMPLETED**: Added training and evaluation examples with parameter documentation
  - [x] ~~Add information about datasets (where to download Stanford-2D-3D-S, Matterport-3D, Structured-3D)~~
    - **COMPLETED**: Added download links for all three datasets
  - [ ] Add link to paper/publication if this is research (will be added when paper is published)
  - [x] ~~Add troubleshooting section~~
    - **DECISION**: Not needed for research project
  - [x] ~~Fix typo in line 17: `$ cd panosamic` → `$ cd PanoSAMic` (matches repo name)~~
    - **COMPLETED**: Fixed to `$ cd PanoSAMic`

- [x] **Missing SAM Weights Instructions**
  - ~~Add where to download SAM weights from~~
  - ~~List which specific weights files are needed~~
  - ~~Document expected filenames (script suggests `sam_vit_h_4b8939.pth`)~~
  - **COMPLETED**: README now includes download link and all three weight file options

- [x] **No Tests**
  - ~~Found cached test file (`__pycache__/test_dual_view_instance.cpython-314.pyc`) but no source~~
  - ~~Add unit tests and integration tests~~
  - **DECISION**: Not needed - this is a research project, tests not required for publication

- [x] **Package Metadata in pyproject.toml**
  - [x] ~~Add: `description`, `keywords`, `homepage`, `repository`, `documentation`~~
    - **COMPLETED**: Added description, keywords, homepage, repository, and issues URLs
  - [x] ~~Add: classifiers (Python versions, license, development status)~~
    - **COMPLETED**: Added Python version classifiers, license, development status, and topic classifiers
  - [x] ~~Add: project URLs (issues, documentation, etc.)~~
    - **COMPLETED**: Added Homepage, Repository, and Issues URLs

## **MEDIUM PRIORITY** (Should Address)

- [x] **Author Email Visibility**
  - ~~Personal DFKI email (`mahdi.chamseddine@dfki.de`) appears in 6 files~~
  - ~~Consider if this should remain or be replaced with a project email~~
  - **COMPLETED**:
    - Removed emails from all file headers (now just "Author: Mahdi Chamseddine")
    - Added author signatures to all 47 Python files in the panosamic directory
    - Email remains in pyproject.toml as project contact point

- [ ] **Incomplete TODOs in Code**
  - [ ] `panosamic/datasets/base.py:260` - `# TODO CHECK`
  - [ ] `panosamic/evaluation/evaluator.py:161` - `# TODO check`
  - [ ] `panosamic/model/panosamic_net.py:70` - `# TODO assert dims`
  - [ ] `panosamic/model/image_encoder.py:124` - `# Memory intensive, TODO alternative?`
  - [ ] Several files mention: `# TODO This will be added in future python versions`

- [x] **Hard-coded Paths in Scripts**
  - ~~`scripts/local_train.sh:3-7` - Contains hard-coded `/data/Datasets/` paths~~
  - ~~`scripts/run_panosamic_eval_multifold.sh:5` - Hard-coded dataset path~~
  - ~~Document these paths or make them more configurable~~
  - **COMPLETED**: All scripts now use environment variable defaults with clear documentation

- [ ] **Missing Examples**
  - [ ] Add example notebooks or scripts showing basic usage
  - [ ] Add visualization examples

- [ ] **Git Ignore Issues**
  - [ ] `.python-version` is tracked but could be in .gitignore (depends on preference)
  - [ ] Clean up `__pycache__` directories from git:
    ```bash
    git rm -r --cached scripts/__pycache__ panosamic/**/__pycache__
    ```

## **LOW PRIORITY** (Nice to Have)

- [ ] **Missing Files**
  - [x] ~~CHANGELOG.md to track version history~~
    - **DECISION**: Not needed for research project
  - [x] ~~CODE_OF_CONDUCT.md~~
    - **DECISION**: Not needed for research project
  - [x] ~~SECURITY.md for reporting vulnerabilities~~
    - **DECISION**: Not needed for research project
  - [ ] GitHub templates (.github/ISSUE_TEMPLATE, PULL_REQUEST_TEMPLATE) - Optional
  - [x] ~~Badges in README (build status, license, Python version, etc.)~~
    - **COMPLETED**: Added Python version and license badges

- [ ] **Documentation**
  - [ ] API documentation or docstring coverage check
  - [ ] Architecture diagram or model overview
  - [ ] Performance benchmarks or comparison tables

- [ ] **Development Setup**
  - [ ] Missing development dependencies specification
  - [ ] No pre-commit hooks configuration
  - [ ] No Dockerfile for reproducible environment

- [ ] **Dependency Specifications**
  - [ ] Dependencies lack version constraints
  - [ ] Example: `"torch"` should be `"torch>=2.0,<3.0"` to avoid compatibility issues

## **VERIFICATION NEEDED**

- [ ] **SAM Weights Directory**
  - Currently empty except for `.gitkeep`
  - Ensure users understand they need to download weights separately
  - Consider adding a download script

- [ ] **Experiments & Visualizations Directories**
  - Listed in `.gitignore:223-224` but don't exist yet
  - This is fine, but users might be confused about where outputs go
  - Consider documenting expected directory structure

## Recommended Action Plan

1. ~~Add a proper LICENSE file (choose appropriate license)~~ ✓ COMPLETED
2. ~~Update `license` field in pyproject.toml~~ ✓ COMPLETED
3. ~~Enhance README with complete instructions and examples~~ ✓ COMPLETED
4. ~~Add CITATION file if this is research code~~ ✓ DECIDED (will add to README when paper published)
5. Review and resolve TODO comments (optional for research project)
6. ~~Add basic tests~~ ✓ DECIDED (not needed for research project)
7. Clean up pycache from git (recommended before publication)
8. ~~Add package metadata to pyproject.toml~~ ✓ COMPLETED
9. ~~Fix hard-coded paths in scripts~~ ✓ COMPLETED

---

**Generated**: 2026-01-10
