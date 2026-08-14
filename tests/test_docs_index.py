from dxlens.docs import count_tokens, render_llms_txt
from dxlens.index import ApiIndex, tokenize


def test_llms_txt_is_compact_and_complete(spec, cfg):
    text = render_llms_txt(spec)
    assert text.startswith("# MiniFFTDx")
    for needle in ("Namespace: `minifftdx`", "`Size<N>()`", "Type<fft_type::c2c|r2c|c2r>()", "is_supported<Desc>::value",
                   "C4: Size must be a power of two"):
        assert needle in text
    assert count_tokens(text) < count_tokens(cfg.header.read_text()) / 2


def test_tokenize_splits_camel_case():
    assert tokenize("ElementsPerThread FFTsPerBlock") == ["elements", "per", "thread", "ffts", "per", "block"]


def test_search_ranks_relevant_items(spec):
    idx = ApiIndex(spec)
    top = lambda q, k=3: [h["name"] for h in idx.search(q, k)]
    assert "FFT::threads_per_fft" in top("how many threads per fft")
    assert "is_supported" in top("is configuration valid")
    assert "Precision" in top("fp32 single precision")
    assert "FFT::execute_batch" in top("batched transforms", 5)
    assert idx.search("zzzzqqq") == []
