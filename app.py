import streamlit as st

from exceptions import TreeClassAppError
from extractors import EXTRACTOR_REGISTRY
from model_wrapper import TreeClassifier
from pipeline import ClassificationPipeline

st.set_page_config(page_title="Tree Classifier — OOP in Action", page_icon="🌳")
st.title("🌳 Tree Classifier — OOP in Action")
st.caption(
    "A tree-image classifier built to double as a live OOP walkthrough "
    "for ML beginners: Interfaces, Inheritance, Polymorphism, Abstract "
    "Classes, Exception Handling, and Multithreading."
)


@st.cache_resource
def load_classifier():
    return TreeClassifier()


try:
    classifier = load_classifier()
    pipeline = ClassificationPipeline(classifier)
    model_ready = True
except TreeClassAppError as e:
    st.error(f"Model failed to load: {e}")
    model_ready = False

uploaded = st.file_uploader("Upload a tree image", type=["jpg", "jpeg", "png"])
method = st.selectbox("Feature extraction method", list(EXTRACTOR_REGISTRY.keys()))
compare_all = st.checkbox("Compare all methods (runs extractors in parallel)")

with st.expander("🔍 What OOP concept is running here?"):
    st.markdown(
        """
- **Interface / Abstract Class** — every extractor inherits from `BaseExtractor`
  and must implement `extract()`.
- **Inheritance** — `EdgeDetectionExtractor`, `ShadowHSLExtractor`, `PCAExtractor`,
  `RawImageExtractor`, `TConnectedExtractor` all share `BaseExtractor`'s
  loading and error-handling logic.
- **Polymorphism** — the dropdown above picks a class at runtime; the code
  calls `.run(image)` without knowing which subclass it's using.
- **Exception Handling** — bad uploads or failed extraction raise custom
  exceptions (`InvalidImageError`, `FeatureExtractionError`) caught here,
  not raw stack traces.
- **Multithreading / Collections** — "Compare all methods" runs every
  extractor concurrently with `ThreadPoolExecutor` and collects results
  in an `OrderedDict`.
        """
    )

if uploaded and model_ready:
    st.image(uploaded, width=300)
    try:
        if compare_all:
            with st.spinner("Running all extractors in parallel..."):
                results = pipeline.run_all(uploaded)
            st.subheader("Results by method")
            st.write(results)
        else:
            with st.spinner(f"Running {method}..."):
                pred = pipeline.run_single(uploaded, method)
            st.success(f"Prediction: {pred}")
    except TreeClassAppError as e:
        st.error(str(e))
