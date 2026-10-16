import unittest

from backend.pdf_refs import build_refs
from backend.pdf_refs.refs_surya import split_surya_reference_blocks


def _text(block_id, text, y):
    return {
        "id": block_id,
        "text": text,
        "x": 10,
        "y": y,
        "w": 500,
        "h": 12,
        "lines": [{"x": 10, "y": y, "w": 500, "h": 12,
                   "runs": [{"t": text}]}],
    }


class ReferenceFallbackTest(unittest.TestCase):
    def test_author_year_tokens_match_entries_without_links(self):
        doc = {
            "parser": "pymupdf",
            "parser_info": {"links_total": 0},
            "pages": [{
                "texts": [
                    _text("p0b0", "See (Smith et al., 2020) and (e.g., [Wu et al., 2021]). "
                          "Not (Wilson and Clark, 2020).", 20),
                    _text("p0b1", "References", 10),
                    _text("p0b2", "[Smith and Jones, 2020] First paper.", 30),
                    _text("p0b3", "[Wu et al., 2021] Second paper.", 50),
                ],
            }],
        }

        result = build_refs(doc)

        # 方括号嵌在圆括号里时只标里层；作者对不上的令牌不标。
        self.assertEqual(
            [(link["token"], link["refs"]) for link in result["links"]],
            [("(Smith et al., 2020)", [0]), ("[Wu et al., 2021]", [1])],
        )

    def test_numeric_citations_map_when_pdf_has_no_goto_links(self):
        doc = {
            "parser": "pymupdf",
            "parser_info": {"links_total": 2},
            "pages": [
                {"texts": [_text("p0b0", "Earlier work [1] and [2, 3] is relevant.", 20)]},
                {"texts": [
                    _text("p1b0", "References", 10),
                    _text("p1b1", "[1] First cited paper, 2020.", 30),
                    _text("p1b2", "[2] Second cited paper, 2021.", 50),
                    _text("p1b3", "[3] Third cited paper, 2022.", 70),
                ]},
            ],
        }

        result = build_refs(doc)

        self.assertEqual(result["counts"]["entries"], 3)
        self.assertEqual(
            [(link["token"], link["nums"], link["refs"]) for link in result["links"]],
            [("[1]", [1], [0]), ("[2, 3]", [2, 3], [1, 2])],
        )

    def test_reference_entries_are_not_marked_as_body_citations(self):
        doc = {
            "parser": "pymupdf",
            "parser_info": {"links_total": 0},
            "pages": [{
                "texts": [
                    _text("p0b0", "References", 10),
                    _text("p0b1", "[1] This entry mentions [2] in its title.", 30),
                    _text("p0b2", "[2] Another cited paper, 2021.", 50),
                ],
            }],
        }

        result = build_refs(doc)

        self.assertEqual(result["links"], [])


class SuryaReferenceTest(unittest.TestCase):
    def test_numbered_ocr_references_become_independent_layout_blocks(self):
        references = _text(
            "p1b1",
            "[1] First paper, 2020.\n[2] Second paper, 2021.",
            30,
        )
        references.update({"h": 40, "size": 10, "kind": "text"})
        references["lines"][0].update({"h": 40, "runs": [
            {"t": "[1] ", "b": True},
            {"t": "First paper, 2020.\n[2] Second paper, 2021."},
        ]})
        pages = [{"texts": [_text("p0b0", "References", 10), references]}]

        added = split_surya_reference_blocks(pages)
        blocks = pages[0]["texts"][1:]

        self.assertEqual(added, {"text": 1})
        self.assertEqual([block["text"] for block in blocks], [
            "[1] First paper, 2020.", "[2] Second paper, 2021.",
        ])
        self.assertEqual(len({block["id"] for block in blocks}), 2)
        self.assertLess(blocks[0]["y"], blocks[1]["y"])
        self.assertTrue(blocks[0]["lines"][0]["runs"][0]["b"])

        result = build_refs({
            "parser": "surya",
            "parser_info": {"links_total": 0},
            "pages": pages,
        })
        self.assertEqual([entry["n"] for entry in result["entries"]], [1, 2])

    def test_splits_multiple_references_from_one_ocr_block(self):
        body = _text("p0b0", "Prior work [35, 2, 5] is relevant.", 20)
        references = _text(
            "p1b1",
            "[1] First paper, 2020.\n"
            "[2] Second paper, 2021.\n"
            "[5] Fifth paper, 2022.\n"
            "[35] Thirty-fifth paper, 2023.",
            30,
        )
        doc = {
            "parser": "surya",
            "parser_info": {"links_total": 18},
            "pages": [
                {"texts": [body]},
                {"texts": [_text("p1b0", "References", 10), references]},
            ],
        }

        result = build_refs(doc)

        self.assertEqual(
            [(entry["n"], entry["text"]) for entry in result["entries"]],
            [
                (1, "[1] First paper, 2020."),
                (2, "[2] Second paper, 2021."),
                (5, "[5] Fifth paper, 2022."),
                (35, "[35] Thirty-fifth paper, 2023."),
            ],
        )
        self.assertEqual(
            [(link["token"], link["nums"], link["refs"])
             for link in result["links"]],
            [("[35, 2, 5]", [2, 5, 35], [1, 2, 3])],
        )

    def test_does_not_mark_citations_inside_surya_reference_blocks(self):
        doc = {
            "parser": "surya",
            "parser_info": {"links_total": 0},
            "pages": [{
                "texts": [
                    _text("p0b0", "References", 10),
                    _text("p0b1", "[1] A paper whose title mentions [2].\n"
                          "[2] Another paper.", 30),
                ],
            }],
        }

        result = build_refs(doc)

        self.assertEqual([entry["n"] for entry in result["entries"]], [1, 2])
        self.assertEqual(result["links"], [])

    def test_surya_keeps_author_year_entries_as_a_separate_fallback(self):
        doc = {
            "parser": "surya",
            "parser_info": {"links_total": 0},
            "pages": [{
                "texts": [
                    _text("p0b0", "References", 10),
                    _text("p0b1", "Smith et al. (2020). First paper.", 30),
                    _text("p0b2", "Jones (2021). Second paper.", 50),
                ],
            }],
        }

        result = build_refs(doc)

        self.assertEqual(result["counts"]["entries"], 2)
        self.assertEqual([entry["n"] for entry in result["entries"]], [None, None])

    def test_author_year_bracket_entries_split_and_map_without_links(self):
        references = _text(
            "p1b1",
            "[Smith and Jones, 2020] First paper.\n"
            "[Wu et al., 2021] Second paper.",
            30,
        )
        references.update({"h": 40, "size": 10, "kind": "text"})
        pages = [{"texts": [
            _text("p0b0", "Prior work [Smith et al., 2020] and [Wu et al., 2021].",
                  20),
            _text("p1b0", "References", 10),
            references,
        ]}]

        added = split_surya_reference_blocks(pages)
        blocks = pages[0]["texts"][2:]

        self.assertEqual(added, {"text": 1})
        self.assertEqual(
            [block["text"] for block in blocks],
            ["[Smith and Jones, 2020] First paper.",
             "[Wu et al., 2021] Second paper."],
        )
        self.assertEqual([block["ref_number"] for block in blocks], [None, None])

        result = build_refs({
            "parser": "surya",
            "parser_info": {"links_total": 0},
            "pages": pages,
        })

        self.assertEqual([entry["n"] for entry in result["entries"]], [None, None])
        self.assertEqual(
            [(link["token"], link["refs"]) for link in result["links"]],
            [("[Smith et al., 2020]", [0]), ("[Wu et al., 2021]", [1])],
        )

    def test_surya_scans_unlinked_numeric_citations_alongside_goto_links(self):
        body = _text("p0b0", "Citations [1] and [2].", 20)
        first_ref = _text("p1b1", "[1] First paper, 2020.", 30)
        second_ref = _text("p1b2", "[2] Second paper, 2021.", 60)
        doc = {
            "parser": "surya",
            "parser_info": {"links_total": 1},
            "pages": [
                {
                    "texts": [body],
                    "links": [{
                        "kind": "goto",
                        "rect": [50, 20, 70, 32],
                        "page": 1,
                        "to": [20, 30],
                    }],
                },
                {"texts": [_text("p1b0", "References", 10), first_ref, second_ref]},
            ],
        }

        result = build_refs(doc)

        self.assertEqual(
            [(link["token"], link["nums"]) for link in result["links"]],
            [("[1]", [1]), ("[2]", [2])],
        )


class UnlimitedOcrReferenceTest(unittest.TestCase):
    def test_reference_kind_fallback_stays_on_unlimited_ocr_strategy(self):
        doc = {
            "parser": "unlimited_ocr",
            "parser_info": {"links_total": 0},
            "pages": [{
                "texts": [
                    _text("p0b0", "Prior work [2].", 10),
                    dict(_text("p0b1", "[1] First paper, 2020.", 30),
                         kind="reference"),
                    dict(_text("p0b2", "[2] Second paper, 2021.", 50),
                         kind="reference"),
                ],
            }],
        }

        result = build_refs(doc)

        self.assertEqual([entry["n"] for entry in result["entries"]], [1, 2])
        self.assertEqual([link["nums"] for link in result["links"]], [[2]])


if __name__ == "__main__":
    unittest.main()
