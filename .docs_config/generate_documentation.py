"""Auto-generate API reference pages."""

import json
from pathlib import Path

import mkdocs_gen_files

# === API Reference ===
src_dirs = [
    ("Shared", Path("src"), "Functions shared between the solvers"),
    ("FEM-only", Path("fem", "src"), "Functions used only in the FEM solver"),
    ("NN-only", Path("nn", "src"), "Functions used only in the neural network solver"),
]

for section, src_dir, description in src_dirs:
    section_nav = mkdocs_gen_files.Nav()

    for path in sorted(src_dir.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        module_path = path.relative_to(src_dir).with_suffix("")
        doc_path = Path("reference", section, path.relative_to(src_dir)).with_suffix(
            ".md"
        )

        parts = tuple(module_path.parts)
        section_nav[parts] = path.relative_to(src_dir).with_suffix(".md").as_posix()

        with mkdocs_gen_files.open(doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}\n")

        mkdocs_gen_files.set_edit_path(doc_path, path)

    with mkdocs_gen_files.open(f"reference/{section}/SUMMARY.md", "w") as nav_file:
        nav_file.write(f"# {section}\n\n")
        nav_file.write(f"{description}\n\n")
        nav_file.writelines(section_nav.build_literate_nav())


# === JSON Schema Documentation ===
def resolve_ref(prop: str, definitions: dict | None) -> dict:
    """Resolve a $ref to its definition."""
    if "$ref" in prop and definitions:
        ref_name = prop["$ref"].split("/")[-1]
        return definitions.get(ref_name, prop)
    return prop


def type_str(prop: dict) -> str:
    """Just the type, no constraints."""
    t = prop.get("type", "object")
    if isinstance(t, list):
        t = " | ".join(t)
    return t


def constraint_str(prop: dict) -> str:
    """Constraint annotations only, empty string if none."""
    extras = []
    if "enum" in prop:
        vals = ", ".join(
            f'"{v}"' if isinstance(v, str) else str(v) for v in prop["enum"]
        )
        extras.append(f"one of: {vals}")
    if "minimum" in prop:
        extras.append(f"min: {prop['minimum']}")
    if "maximum" in prop:
        extras.append(f"max: {prop['maximum']}")
    if "minItems" in prop:
        extras.append(f"min items: {prop['minItems']}")
    if "maxItems" in prop:
        extras.append(f"max items: {prop['maxItems']}")
    return ", ".join(extras)


DEPTH_COLORS = ["#448aff", "#69f0ae", "#ffd740", "#ff6b6b"]


def render_schema_table(
    properties: dict, definitions: list = None, depth: int = 0
) -> list:
    """Recursively render a schema properties dict as a list of markdown table rows.

    Each row is a string of the form "| field | type | description |". Nested fields are rendered with indentation and colour coding to indicate depth.

    Args:
        properties: The "properties" dict from a JSON schema.
        definitions: The "definitions" dict from a JSON schema, used to resolve $ref references.
        depth: The current depth of recursion, used for colour coding and indentation.

    Returns:
        A list of strings, each representing a row in the markdown table.
    """
    rows = []
    for name, prop in properties.items():
        prop = resolve_ref(prop, definitions)
        t = type_str(prop)
        constraints = constraint_str(prop)
        desc = prop.get("description", "—")
        if constraints:
            badge = (
                f'<span style="background:var(--md-code-bg-color);'
                f'border-radius:3px;padding:1px 6px;font-size:0.85em;font-weight:600">'
                f"{constraints}</span>"
            )
            desc = f"{desc}<br>{badge}" if desc != "—" else badge

        nested_props = prop.get("properties")
        items = prop.get("items")
        if isinstance(items, dict):
            items = resolve_ref(items, definitions)

        color = DEPTH_COLORS[min(depth, len(DEPTH_COLORS) - 1)]
        margin = depth * 16
        field_cell = (
            f'<span style="border-left:3px solid {color};'
            f"margin-left:{margin}px;"
            f"padding-left:8px;"
            f'display:inline-block">'
            f"`{name}`</span>"
        )

        rows.append(f"| {field_cell} | `{t}` | {desc} |")

        if nested_props:
            rows.extend(render_schema_table(nested_props, definitions, depth + 1))
        elif isinstance(items, dict) and "properties" in items:
            rows.extend(
                render_schema_table(items["properties"], definitions, depth + 1)
            )

    return rows


def render_depth_legend() -> str:
    """Render a legend explaining the colour coding for field nesting depth.

    Returns:
        A string of HTML representing the legend, which can be included in the markdown documentation.
    """
    legend_items = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;padding:4px 0; font-size:0.8em">'
        f'<span style="width:16px;height:16px;border-radius:3px;background:{color};flex-shrink:0;display:inline-block"></span>'
        f'<span style="font-weight:500">Depth {i}</span>'
        f'<span style="color:var(--md-default-fg-color--light)">{["Top-level field", "Level 1 child", "Level 2 child", "Level 3 child"][i]}</span>'
        f"</div>"
        for i, color in enumerate(DEPTH_COLORS)
    )

    tree = (
        "root field            (depth 0)\n"
        "├─ child field        (depth 1)\n"
        "│   ├─ grandchild     (depth 2)\n"
        "│   └─ grandchild     (depth 2)\n"
        "└─ child field        (depth 1)"
    )

    return (
        '<div style="border:1px solid var(--md-default-fg-color--lightest);border-radius:4px;padding:12px 16px;margin-bottom:16px">'
        "<strong>Field nesting depth</strong><br>"
        "Colours indicate nesting level — the deeper the field, the further right it appears."
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:32px;margin-top:8px;align-items:center">'
        f'<div style="font-family:monospace;font-size:0.85em;white-space:pre;line-height:1.6">{tree}</div>'
        f'<div style="display:flex;flex-direction:column;justify-content:center">{legend_items}</div>'
        "</div>"
        "</div>\n\n"
    )


repo_root = Path(__file__).resolve().parent.parent

schemas = [
    ("Parent Input File", "parent.json"),
    ("FEM Input File", "fem.json"),
    ("NN Input File", "nn.json"),
]

schema_dir = repo_root / "input" / "schemas"
schema_nav = mkdocs_gen_files.Nav()

for title, filename in schemas:
    full_path = schema_dir / filename
    if not full_path.exists():
        continue

    with open(full_path) as f:
        schema = json.load(f)

    slug = filename.replace(".json", "")
    doc_path = Path("reference", "schemas", f"{slug}.md")

    definitions = schema.get("definitions", {})
    properties = schema.get("properties", {})
    description = schema.get("description", "")

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(f"# {title}\n\n")
        if description:
            fd.write(f"*{description}*\n\n")
        fd.write(render_depth_legend())
        fd.write("| Field | Type | Description |\n")
        fd.write("|---|---|---|\n")
        fd.write("\n".join(render_schema_table(properties, definitions)))
        fd.write("\n")

    schema_nav[(title,)] = f"{slug}.md"

with mkdocs_gen_files.open("reference/schemas/SUMMARY.md", "w") as nav_file:
    nav_file.write("# Input File Schemas\n\n")
    nav_file.write("Auto-generated documentation for the JSON input file schemas.\n\n")
    nav_file.writelines(schema_nav.build_literate_nav())
