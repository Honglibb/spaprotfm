from spaprotfm.data.markers import (
    canonicalize,
    load_alias_table,
    standardize_panel,
)


def test_canonicalize_strips_whitespace_and_lowers():
    assert canonicalize("  CD8a  ") == "cd8a"


def test_canonicalize_removes_greek_variants():
    assert canonicalize("CD8α") == "cd8a"
    assert canonicalize("HLA-DRβ") == "hla-drb"


def test_canonicalize_normalizes_dashes():
    assert canonicalize("HLA_DR") == "hla-dr"
    assert canonicalize("HLA DR") == "hla-dr"


def test_load_alias_table_returns_dict_of_canonical_to_aliases(tmp_path):
    yaml_path = tmp_path / "aliases.yaml"
    yaml_path.write_text(
        "CD8A:\n  - CD8\n  - CD8a\n  - CD8α\n"
        "PAN-CK:\n  - PanCK\n  - Pan-Keratin\n"
    )
    table = load_alias_table(yaml_path)
    assert table["cd8a"] == "CD8A"
    assert table["cd8"] == "CD8A"
    assert table["pan-ck"] == "PAN-CK"
    assert table["panck"] == "PAN-CK"
    assert table["pan-keratin"] == "PAN-CK"


def test_standardize_panel_maps_known_and_warns_unknown(tmp_path, caplog):
    yaml_path = tmp_path / "aliases.yaml"
    yaml_path.write_text("CD8A:\n  - CD8a\n")
    table = load_alias_table(yaml_path)

    out = standardize_panel(["CD8a", "FOXP3"], table)
    assert out == ["CD8A", None]
    assert "FOXP3" in caplog.text
