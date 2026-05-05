from __future__ import annotations

"""
Generate a project-level code map from local Python source files.

This script scans the Python modules in `src/`, `scripts/`, and `tests/`,
extracts top-level function definitions with the `ast` module, and records
direct project-internal function calls between them. The current project layout
includes a dedicated `src/diagnostics.py` layer, so the generated maps now make
the root-diagnostic orchestration visible instead of folding it implicitly into
the experiment runners.

The output is intentionally split into several artifacts under
`docs/code_map/`:
- A Markdown overview with Mermaid diagrams and per-module call tables
- A module-level Graphviz `.dot` file and rendered `.svg`
- A solver-core Graphviz `.dot` file and rendered `.svg`
- An experiments/tests Graphviz `.dot` file and rendered `.svg`

This file is not part of the numerical solver itself. It is a documentation
and code-reading utility for the repository. The goal is to make the project
structure legible to a reviewer or future maintainer without requiring them to
manually trace imports and calls across every module.

Important scope choice:
The analysis records only direct calls whose target can be identified from a
local function name in the parsed files. It does not infer dynamic dispatch or
indirect calls through function arguments such as `potential_fn(...)`. That
restriction keeps the tool simple and deterministic, while still recovering
the main control flow of the project.
"""

import ast
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_MAP_DIR = PROJECT_ROOT / "docs" / "code_map"
OUTPUT_PATH = CODE_MAP_DIR / "code_map.md"
MODULE_DOT_PATH = CODE_MAP_DIR / "code_map_modules.dot"
MODULE_SVG_PATH = CODE_MAP_DIR / "code_map_modules.svg"
CORE_DOT_PATH = CODE_MAP_DIR / "code_map_core.dot"
CORE_SVG_PATH = CODE_MAP_DIR / "code_map_core.svg"
EXPERIMENTS_DOT_PATH = CODE_MAP_DIR / "code_map_experiments.dot"
EXPERIMENTS_SVG_PATH = CODE_MAP_DIR / "code_map_experiments.svg"
SOURCE_DIRS = ("src", "scripts", "tests")


# ===========================================================================
# DATA CLASS: PotentialMapSpec
# ===========================================================================
@dataclass(frozen=True)
class PotentialMapSpec:
    """
    Specification for one potential-specific code-map view.

    Attributes
    ----------
    slug : str
        Short filesystem-safe identifier used in output filenames.
    title : str
        Human-readable graph title.
    root_functions : tuple[str, ...]
        Experiment entry points whose reachable call graph defines the map.
    allowed_functions : frozenset[str] or None
        Optional filter used to keep only a chosen subset of the reachable
        functions. This is mainly needed for `run_scattering()`, where one
        experiment function contains both the single- and double-barrier flows.
    """

    slug: str
    title: str
    root_functions: tuple[str, ...]
    allowed_functions: frozenset[str] | None = None


POTENTIAL_MAP_SPECS = (
    PotentialMapSpec(
        slug="infinite_square_well",
        title="Infinite Square Well Experiment",
        root_functions=("src.experiments.run_square_well",),
    ),
    PotentialMapSpec(
        slug="harmonic_oscillator",
        title="Harmonic Oscillator Experiment",
        root_functions=("src.experiments.run_harmonic_oscillator",),
    ),
    PotentialMapSpec(
        slug="quartic_double_well",
        title="Quartic Double-Well Experiment",
        root_functions=("src.experiments.run_quartic_double_well",),
    ),
    PotentialMapSpec(
        slug="finite_square_well",
        title="Finite Square Well Experiment",
        root_functions=("src.experiments.run_finite_square_well",),
    ),
    PotentialMapSpec(
        slug="single_square_barrier",
        title="Single Square-Barrier Scattering Experiment",
        root_functions=("src.experiments.run_scattering",),
        allowed_functions=frozenset(
            {
                "src.experiments.run_scattering",
                "src.experiments._experiment_results_dir",
                "src.potentials.square_barrier",
                "src.analysis.save_csv_rows",
                "src.plotting._ensure_parent",
                "src.plotting.plot_scattering_coefficients",
                "src.scattering.numerov_outward_complex",
                "src.scattering.integrate_from_right",
                "src.scattering.decompose_left_asymptotic",
                "src.scattering.solve_scattering",
                "src.scattering.sweep_scattering",
                "src.numerov.q_from_energy",
            }
        ),
    ),
    PotentialMapSpec(
        slug="double_square_barrier",
        title="Double Square-Barrier Scattering Experiment",
        root_functions=("src.experiments.run_scattering",),
        allowed_functions=frozenset(
            {
                "src.experiments.run_scattering",
                "src.experiments._experiment_results_dir",
                "src.potentials.double_square_barrier",
                "src.analysis.save_csv_rows",
                "src.plotting._ensure_parent",
                "src.plotting.plot_scattering_coefficients",
                "src.plotting.plot_scattering_potential_and_probability",
                "src.scattering.numerov_outward_complex",
                "src.scattering.integrate_from_right",
                "src.scattering.decompose_left_asymptotic",
                "src.scattering.solve_scattering",
                "src.scattering.scattering_wavefunction",
                "src.scattering.sweep_scattering",
                "src.scattering.find_transmission_peaks",
                "src.numerov.q_from_energy",
            }
        ),
    ),
)


# ===========================================================================
# FUNCTION: project_python_files
# ===========================================================================
def project_python_files() -> list[Path]:
    """
    Collect the Python files that should be included in the code map.

    Returns
    -------
    list[Path]
        Sorted list of top-level Python files from the configured source
        directories, excluding this generator script itself so the map
        describes the project rather than the documentation helper.
    """

    files: list[Path] = []
    for directory in SOURCE_DIRS:
        files.extend(sorted((PROJECT_ROOT / directory).glob("*.py")))

    # Exclude this script so the generated map stays focused on the project
    # code rather than recursively documenting the mapper.
    return [path for path in files if path.name != "generate_code_map.py"]


# ===========================================================================
# FUNCTION: module_name
# ===========================================================================
def module_name(path: Path) -> str:
    """
    Convert a repository-relative file path to a dotted module name.

    Parameters
    ----------
    path : Path
        Python file inside the project tree.

    Returns
    -------
    str
        Dotted module path such as ``src.shooting`` or
        ``tests.test_solver``.
    """

    rel = path.relative_to(PROJECT_ROOT).with_suffix("")

    return ".".join(rel.parts)


# ===========================================================================
# FUNCTION: short_name
# ===========================================================================
def short_name(qualified_name: str) -> str:
    """
    Return the final name segment of a qualified symbol.

    Parameters
    ----------
    qualified_name : str
        Fully qualified function or module-qualified symbol name.

    Returns
    -------
    str
        Unqualified final component, used for compact graph labels.
    """

    return qualified_name.split(".")[-1]


# ===========================================================================
# FUNCTION: file_key
# ===========================================================================
def file_key(qualified_name: str) -> str:
    """
    Reduce a qualified function name to its module key.

    Parameters
    ----------
    qualified_name : str
        Fully qualified function name such as ``src.shooting.main``.

    Returns
    -------
    str
        Module-level key consisting of the first two dotted components, for
        example ``src.shooting`` or ``tests.test_solver``.
    """

    parts = qualified_name.split(".")

    return ".".join(parts[:2])


# ===========================================================================
# FUNCTION: build_function_index
# ===========================================================================
def build_function_index(files: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    """
    Build lookup tables for top-level project functions.

    The first mapping lets the call collector resolve a simple function name to
    its fully qualified project name. The second mapping records which module
    owns each qualified function.

    Parameters
    ----------
    files : list[Path]
        Python source files to parse.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        Pair ``(qualified_by_local_name, module_by_function)`` used later when
        constructing the call graph.
    """

    qualified_by_local_name: dict[str, str] = {}
    module_by_function: dict[str, str] = {}

    for path in files:
        module = module_name(path)
        tree = ast.parse(path.read_text(), filename=str(path))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{module}.{node.name}"
                qualified_by_local_name[node.name] = qualified
                module_by_function[qualified] = module

    return qualified_by_local_name, module_by_function


# ===========================================================================
# CLASS: CallCollector
# ===========================================================================
class CallCollector(ast.NodeVisitor):
    """
    AST visitor that collects direct calls to known project functions.

    The visitor inspects each `ast.Call` node and records the target when it
    can match either a bare function name or the attribute name of a method-like
    access against the indexed set of local project functions.
    """

    def __init__(self, qualified_by_local_name: dict[str, str]) -> None:
        """
        Initialize the collector with the known project function index.

        Parameters
        ----------
        qualified_by_local_name : dict[str, str]
            Mapping from simple function names to fully qualified names.
        """

        self.qualified_by_local_name = qualified_by_local_name
        self.calls: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        """
        Record the call target if it resolves to a known project function.

        Parameters
        ----------
        node : ast.Call
            AST call expression being visited.

        Returns
        -------
        None
        """

        called_name: str | None = None

        if isinstance(node.func, ast.Name):
            called_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called_name = node.func.attr

        # This deliberately matches only by name, not by full import graph
        # resolution. The resulting map is conservative and easy to reproduce.
        if called_name in self.qualified_by_local_name:
            self.calls.add(self.qualified_by_local_name[called_name])

        self.generic_visit(node)


# ===========================================================================
# FUNCTION: build_call_graph
# ===========================================================================
def build_call_graph(
    files: list[Path], qualified_by_local_name: dict[str, str]
) -> dict[str, list[str]]:
    """
    Build the direct function-level call graph for the project.

    Parameters
    ----------
    files : list[Path]
        Python files to analyze.
    qualified_by_local_name : dict[str, str]
        Mapping from simple function names to fully qualified project names.

    Returns
    -------
    dict[str, list[str]]
        Mapping from each fully qualified function name to the sorted list of
        directly called project functions found in its body.
    """

    graph: dict[str, list[str]] = {}

    for path in files:
        module = module_name(path)
        tree = ast.parse(path.read_text(), filename=str(path))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                collector = CallCollector(qualified_by_local_name)
                collector.visit(node)
                graph[f"{module}.{node.name}"] = sorted(collector.calls)

    return graph


# ===========================================================================
# FUNCTION: module_edges
# ===========================================================================
def module_edges(graph: dict[str, list[str]]) -> set[tuple[str, str]]:
    """
    Collapse the function-level graph into inter-module edges.

    Parameters
    ----------
    graph : dict[str, list[str]]
        Direct function call graph keyed by qualified function name.

    Returns
    -------
    set[tuple[str, str]]
        Unique directed edges ``(caller_module, callee_module)`` excluding
        self-edges within the same module.
    """

    edges: set[tuple[str, str]] = set()
    for caller, callees in graph.items():
        caller_module = file_key(caller)

        for callee in callees:
            callee_module = file_key(callee)
            if caller_module != callee_module:
                edges.add((caller_module, callee_module))

    return edges


# ===========================================================================
# FUNCTION: mermaid_id
# ===========================================================================
def mermaid_id(name: str) -> str:
    """
    Sanitize a name for use as a Mermaid node identifier.

    Parameters
    ----------
    name : str
        Module or function identifier containing dots.

    Returns
    -------
    str
        Mermaid-safe identifier with dots replaced by underscores.
    """

    return name.replace(".", "_")


# ===========================================================================
# FUNCTION: render_module_graph
# ===========================================================================
def render_module_graph(edges: set[tuple[str, str]]) -> list[str]:
    """
    Render the module-level flow graph as Mermaid Markdown lines.

    Parameters
    ----------
    edges : set[tuple[str, str]]
        Directed edges between modules.

    Returns
    -------
    list[str]
        Mermaid code block lines for embedding in Markdown.
    """

    lines = ["```mermaid", "flowchart LR"]
    for caller, callee in sorted(edges):
        lines.append(
            f"    {mermaid_id(caller)}[{caller}] --> {mermaid_id(callee)}[{callee}]"
        )

    lines.append("```")

    return lines


# ===========================================================================
# FUNCTION: reachable_functions
# ===========================================================================
def reachable_functions(
    graph: dict[str, list[str]], root_functions: tuple[str, ...]
) -> set[str]:
    """
    Compute the set of functions reachable from one or more graph roots.

    Parameters
    ----------
    graph : dict[str, list[str]]
        Direct project function call graph.
    root_functions : tuple[str, ...]
        Fully qualified root functions from which traversal begins.

    Returns
    -------
    set[str]
        Reachable functions including the supplied roots.
    """

    visited: set[str] = set()
    stack = [root for root in root_functions if root in graph]

    while stack:
        current = stack.pop()

        if current in visited:
            continue

        visited.add(current)
        stack.extend(
            callee for callee in graph.get(current, []) if callee not in visited
        )

    return visited


# ===========================================================================
# FUNCTION: functions_for_spec
# ===========================================================================
def functions_for_spec(graph: dict[str, list[str]], spec: PotentialMapSpec) -> set[str]:
    """
    Resolve the concrete function set for one potential-specific map spec.

    Parameters
    ----------
    graph : dict[str, list[str]]
        Direct project function call graph.
    spec : PotentialMapSpec
        Potential-specific graph specification.

    Returns
    -------
    set[str]
        Selected function set to render for the requested potential.
    """

    included = reachable_functions(graph, spec.root_functions)

    if spec.allowed_functions is not None:
        included &= spec.allowed_functions

    return included


# ===========================================================================
# FUNCTION: render_selected_focus_graph
# ===========================================================================
def render_selected_focus_graph(
    title: str,
    graph: dict[str, list[str]],
    included_functions: set[str],
) -> list[str]:
    """
    Render a filtered function-level Mermaid graph for selected functions.

    Parameters
    ----------
    title : str
        Section title placed above the Mermaid block.
    graph : dict[str, list[str]]
        Full direct function call graph.
    included_functions : set[str]
        Exact set of fully qualified functions to render.

    Returns
    -------
    list[str]
        Markdown lines containing a titled Mermaid graph.
    """

    lines = [f"## {title}", "", "```mermaid", "flowchart TD"]

    for caller in sorted(included_functions):
        node_id = mermaid_id(caller)
        lines.append(f"    {node_id}[{short_name(caller)}]")

        for callee in graph[caller]:
            if callee in included_functions:
                lines.append(
                    f"    {node_id} --> {mermaid_id(callee)}[{short_name(callee)}]"
                )

    lines.append("```")

    return lines


# ===========================================================================
# FUNCTION: render_focus_graph
# ===========================================================================
def render_focus_graph(
    title: str,
    graph: dict[str, list[str]],
    include_modules: set[str],
) -> list[str]:
    """
    Render a filtered function-level Mermaid graph for selected modules.

    Parameters
    ----------
    title : str
        Section title placed above the Mermaid block.
    graph : dict[str, list[str]]
        Full direct function call graph.
    include_modules : set[str]
        Module keys whose functions should be included in the focused view.

    Returns
    -------
    list[str]
        Markdown lines containing a titled Mermaid graph.
    """

    included_functions = {func for func in graph if file_key(func) in include_modules}

    return render_selected_focus_graph(title, graph, included_functions)


# ===========================================================================
# FUNCTION: render_call_tables
# ===========================================================================
def render_call_tables(graph: dict[str, list[str]]) -> list[str]:
    """
    Render a readable per-module summary of direct function calls.

    Parameters
    ----------
    graph : dict[str, list[str]]
        Full direct function call graph.

    Returns
    -------
    list[str]
        Markdown lines listing ``caller -> callees`` by module.
    """

    by_module: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)

    for caller, callees in sorted(graph.items()):
        by_module[file_key(caller)].append(
            (short_name(caller), [short_name(c) for c in callees])
        )

    lines = ["## Direct Function Calls", ""]

    for module in sorted(by_module):
        lines.append(f"### {module}")
        lines.append("")

        for caller, callees in by_module[module]:
            if callees:
                joined = ", ".join(callees)
            else:
                joined = "(no direct project-function calls)"

            lines.append(f"- `{caller}` -> {joined}")

        lines.append("")

    return lines


# ===========================================================================
# FUNCTION: dot_id
# ===========================================================================
def dot_id(name: str) -> str:
    """
    Sanitize a name for use as a Graphviz node identifier.

    Parameters
    ----------
    name : str
        Module or function identifier containing dots.

    Returns
    -------
    str
        Graphviz-safe identifier with dots replaced by underscores.
    """

    return name.replace(".", "_")


# ===========================================================================
# FUNCTION: render_dot_graph
# ===========================================================================
def render_dot_graph(
    title: str,
    graph: dict[str, list[str]],
    include_modules: set[str],
) -> str:
    """
    Render a filtered function-level Graphviz DOT graph.

    Parameters
    ----------
    title : str
        Graph title.
    graph : dict[str, list[str]]
        Full direct function call graph.
    include_modules : set[str]
        Module keys whose functions should appear in the rendered graph.

    Returns
    -------
    str
        Complete DOT source string.
    """

    included_functions = {func for func in graph if file_key(func) in include_modules}

    return render_selected_dot_graph(title, graph, included_functions)


# ===========================================================================
# FUNCTION: render_selected_dot_graph
# ===========================================================================
def render_selected_dot_graph(
    title: str,
    graph: dict[str, list[str]],
    included_functions: set[str],
) -> str:
    """
    Render a filtered function-level Graphviz DOT graph for explicit nodes.

    Parameters
    ----------
    title : str
        Graph title.
    graph : dict[str, list[str]]
        Full direct function call graph.
    included_functions : set[str]
        Exact set of fully qualified functions to render.

    Returns
    -------
    str
        Complete DOT source string.
    """

    lines = [
        "digraph G {",
        '  rankdir="TB";',
        '  graph [labeljust="l", labelloc="t", fontsize=20, fontname="Helvetica"];',
        f'  label="{title}";',
        '  node [shape=box, style="rounded,filled", fillcolor="#F8FAFC", color="#334155", fontname="Helvetica", fontsize=11];',
        '  edge [color="#64748B", arrowsize=0.7];',
    ]

    for caller in sorted(included_functions):
        lines.append(f'  {dot_id(caller)} [label="{short_name(caller)}"];')

    for caller in sorted(included_functions):
        for callee in graph[caller]:
            if callee in included_functions:
                lines.append(f"  {dot_id(caller)} -> {dot_id(callee)};")

    lines.append("}")

    return "\n".join(lines) + "\n"


# ===========================================================================
# FUNCTION: render_module_dot
# ===========================================================================
def render_module_dot(edges: set[tuple[str, str]]) -> str:
    """
    Render the module-level Graphviz DOT graph.

    Parameters
    ----------
    edges : set[tuple[str, str]]
        Directed edges between project modules.

    Returns
    -------
    str
        Complete DOT source string for the module overview.
    """

    lines = [
        "digraph G {",
        '  rankdir="LR";',
        '  graph [labeljust="l", labelloc="t", fontsize=20, fontname="Helvetica"];',
        '  label="Module-Level Flow";',
        '  node [shape=box, style="rounded,filled", fillcolor="#EFF6FF", color="#1D4ED8", fontname="Helvetica", fontsize=12];',
        '  edge [color="#64748B", arrowsize=0.8];',
    ]

    modules = sorted({module for edge in edges for module in edge})

    for module in modules:
        lines.append(f'  {dot_id(module)} [label="{module}"];')

    for caller, callee in sorted(edges):
        lines.append(f"  {dot_id(caller)} -> {dot_id(callee)};")

    lines.append("}")

    return "\n".join(lines) + "\n"


# ===========================================================================
# FUNCTION: render_svg
# ===========================================================================
def render_svg(dot_path: Path, svg_path: Path) -> None:
    """
    Render a Graphviz DOT file to SVG if the `dot` executable is available.

    Parameters
    ----------
    dot_path : Path
        Input DOT file path.
    svg_path : Path
        Output SVG file path.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If Graphviz `dot` is not available on the local system.
    CalledProcessError
        If `dot` fails while rendering the SVG.
    """

    dot_executable = shutil.which("dot")

    if dot_executable is None:
        raise RuntimeError(
            "Graphviz `dot` was not found on PATH, so SVG code maps cannot be rendered."
        )

    subprocess.run(
        [dot_executable, "-Tsvg", str(dot_path), "-o", str(svg_path)],
        check=True,
    )


# ===========================================================================
# FUNCTION: build_markdown
# ===========================================================================
def build_markdown(graph: dict[str, list[str]]) -> str:
    """
    Build the main Markdown code-map document.

    Parameters
    ----------
    graph : dict[str, list[str]]
        Full direct function call graph for the analyzed project files.

    Returns
    -------
    str
        Markdown document containing counts, links, Mermaid diagrams, and
        per-module direct call tables.
    """

    total_functions = len(graph)
    total_edges = sum(len(callees) for callees in graph.values())

    lines = [
        "# Code Map",
        "",
        "Generated from AST analysis of the local Python files in `src/`, `scripts/`, and `tests/`.",
        "",
        f"- Functions mapped: `{total_functions}`",
        f"- Direct project-call edges: `{total_edges}`",
        "- Scope: direct calls only; indirect calls through function arguments such as `potential_fn(...)` are not inferred.",
        "- Regenerate with: `python3 scripts/generate_code_map.py`",
        "- Rendered diagrams:",
        f"- [Module flow SVG]({MODULE_SVG_PATH}:1)",
        f"- [Core solver SVG]({CORE_SVG_PATH}:1)",
        f"- [Experiments and tests SVG]({EXPERIMENTS_SVG_PATH}:1)",
        "",
        "## Potential-Specific Maps",
        "",
    ]

    for spec in POTENTIAL_MAP_SPECS:
        lines.extend(
            [
                f"### {spec.title}",
                "",
                f"- [DOT]({CODE_MAP_DIR / f'code_map_{spec.slug}.dot'}:1)",
                f"- [SVG]({CODE_MAP_DIR / f'code_map_{spec.slug}.svg'}:1)",
                "",
            ]
        )

    lines.extend(
        [
            "## Module-Level Flow",
            "",
        ]
    )
    lines.extend(render_module_graph(module_edges(graph)))
    lines.append("")
    lines.extend(
        render_focus_graph(
            "Bound-State and Scattering Core",
            graph,
            {
                "src.numerov",
                "src.shooting",
                "src.scattering",
                "src.RK4_harmonic_oscillator",
            },
        )
    )
    lines.append("")
    lines.extend(
        render_focus_graph(
            "Experiments and Entry Points",
            graph,
            {
                "src.analysis",
                "src.plotting",
                "src.experiments",
                "scripts.run_solver",
                "tests.test_solver",
            },
        )
    )
    lines.append("")

    for spec in POTENTIAL_MAP_SPECS:
        lines.extend(
            render_selected_focus_graph(
                spec.title,
                graph,
                functions_for_spec(graph, spec),
            )
        )
        lines.append("")

    lines.extend(render_call_tables(graph))

    return "\n".join(lines).rstrip() + "\n"


# ===========================================================================
# FUNCTION: main
# ===========================================================================
def main() -> None:
    """
    Generate the Markdown and Graphviz code-map artifacts.

    The script writes the textual map plus the three DOT files that are later
    rendered to SVG with `dot`.

    Returns
    -------
    None
    """

    files = project_python_files()
    qualified_by_local_name, _ = build_function_index(files)
    graph = build_call_graph(files, qualified_by_local_name)

    CODE_MAP_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(build_markdown(graph))
    MODULE_DOT_PATH.write_text(render_module_dot(module_edges(graph)))
    CORE_DOT_PATH.write_text(
        render_dot_graph(
            "Bound-State and Scattering Core",
            graph,
            {
                "src.numerov",
                "src.shooting",
                "src.scattering",
                "src.RK4_harmonic_oscillator",
            },
        )
    )
    EXPERIMENTS_DOT_PATH.write_text(
        render_dot_graph(
            "Experiments and Entry Points",
            graph,
            {
                "src.analysis",
                "src.plotting",
                "src.experiments",
                "scripts.run_solver",
                "tests.test_solver",
            },
        )
    )

    for spec in POTENTIAL_MAP_SPECS:
        dot_path = CODE_MAP_DIR / f"code_map_{spec.slug}.dot"
        svg_path = CODE_MAP_DIR / f"code_map_{spec.slug}.svg"
        dot_path.write_text(
            render_selected_dot_graph(
                spec.title,
                graph,
                functions_for_spec(graph, spec),
            )
        )
        render_svg(dot_path, svg_path)
    render_svg(MODULE_DOT_PATH, MODULE_SVG_PATH)
    render_svg(CORE_DOT_PATH, CORE_SVG_PATH)
    render_svg(EXPERIMENTS_DOT_PATH, EXPERIMENTS_SVG_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
