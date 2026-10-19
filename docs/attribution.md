# Dependencies and reference attribution

This records the licence evidence inspected on 19 October 2026. It does not
grant rights to third-party material or choose a licence for BlackJackAI.
The detailed `dependency_licenses.json` inventory is preserved in the verified
pre-refinement archive. It records original package metadata, relative licence
locations and SHA-256 checksums; licence paths are relative to the original
installed `site-packages/`. The environment itself is excluded from Git.

## Python dependencies

| Dependency inspected | Use | Licence evidence |
| --- | --- | --- |
| PyDealer 1.4.0, Alex Crawford | Card, deck and stack objects used by the simulator | Installed metadata and source headers say `GPLv3`; the distribution includes the GNU GPL version 3 text. [Upstream source](https://github.com/Trebek/pydealer), [licence](https://github.com/Trebek/pydealer/blob/master/LICENSE.txt). |
| PyTorch 2.14.1 | DQN training components and frozen-policy inference | Installed metadata gives a composite SPDX expression covering several licences; the archived inventory preserves the exact expression. The wheel includes a main licence and additional third-party notices. [Upstream source](https://github.com/pytorch/pytorch), [licence](https://github.com/pytorch/pytorch/blob/main/LICENSE). |
| NumPy 2.5.4 | Numerical dependency available in the audited environment | Installed metadata gives a composite SPDX expression. Its main licence includes the NumPy BSD terms and bundled-library notices. [Upstream source](https://github.com/numpy/numpy), [licence](https://github.com/numpy/numpy/blob/main/LICENSE.txt). |
| pytest 9.1.1 | Automated tests | Installed metadata says `MIT`; the licence credits Holger Krekel and others. [Upstream source](https://github.com/pytest-dev/pytest), [licence](https://github.com/pytest-dev/pytest/blob/main/LICENSE). |
| Matplotlib 3.11.2 | Notebook figures | The installed `License` field contains the Matplotlib licence agreement and bundled-component notices; it has no `License-Expression` field. The full field and installed licence checksum are recorded in the archived inventory. [Upstream source](https://github.com/matplotlib/matplotlib), [licence](https://matplotlib.org/stable/project/license.html). |

The first four packages were inspected in the original working environment.
All five were also inspected in the earlier fresh isolated installation, where
Matplotlib was added for plotting. Notebook execution uses the upstream
[nbformat](https://github.com/jupyter/nbformat),
[nbclient](https://github.com/jupyter/nbclient) and
[ipykernel](https://github.com/ipython/ipykernel) packages; their distributions
retain their own licence texts and notices. The final
requirements and installation verification are documented in
[reproducibility.md](reproducibility.md). Wheel contents may include additional
third-party components; the main licence checksum is not an exhaustive list of
their notices. Retain the applicable licence texts and notices when distributing
dependency source or binaries, and review the actual distribution being shipped.

The archived inventory also covers the fresh environment's transitive
dependencies and packaging tools, including their unaltered licence metadata and the checksums of
licence and notice files listed in wheel metadata. This is an evidence inventory,
not a determination that every possible distribution is compliant. It does not
replace the licence texts in the installed packages. Matplotlib's bundled font
and other component notices, and PyTorch's bundled component notices, have their
own scope; a package's short licence label must not erase those terms.

**PyDealer remains a publication review item.** The project imports its classes
in the simulator. The available evidence does not settle the licence compatibility
and distribution obligations of the combined project. Excluding `.venv/` from Git
does not by itself resolve that question. No project licence has been created or
assumed; choose an appropriate distribution approach after reviewing the GPL
requirements and the project's ownership and intended licence.

## Wizard of Odds reference fixture

The independent reference fixture is
[`tests/fixtures/wizard_s17_das.json`](../tests/fixtures/wizard_s17_das.json).
Its recorded source is the
[Wizard of Odds blackjack strategy calculator](https://wizardofodds.com/games/blackjack/strategy/calculator/),
checked on 18 October 2026. The recorded settings are four-or-more decks
(including six), dealer stands on soft 17, doubling after splitting allowed,
no surrender, and dealer peek for blackjack. Its provenance says decision tokens
were extracted from the rendered calculator, independently of project constants.
The fixture and its independent validation tests have been retained unchanged.

**Publication blocker: permission to redistribute this fixture is unresolved.**
The inspected calculator carries an all-rights-reserved notice. The inspected
[site terms](https://wizardofodds.com/terms-conditions/) do not provide an explicit
fixture redistribution grant, and no permission evidence was found in the
project. This is an unresolved rights question, not a conclusion about the
copyright status of mathematical facts. Attribution alone does not establish
permission. Before public distribution, obtain permission or an appropriate rights
determination, or replace the fixture with a separately sourced, independently
licensed reference and preserve meaningful chart-validation coverage. No contact
has been sent on the author's behalf.

Keep the recorded provenance when resolving this issue. Do not replace the
independent fixture with values generated from the implementation and describe
that as independent validation.

When the owner later removes `docs/`, transfer applicable attribution and notices
to a retained location such as `NOTICE.md`. The current source-only tree does not
bundle dependency distributions; this document is attribution and review
guidance, not a grant to redistribute them.
