"use client";

import { FormEvent, useState } from "react";

type Sentiment = "Positive" | "Negative" | "Neutral";
type Classification = "Genuine" | "Fake / Deceptive";

type AspectResult = {
  aspect: string;
  sentiment: Sentiment;
  score: number;
};

type AnalysisResult = {
  review: string;
  classification: Classification;
  confidence: number | null;
  aspects: AspectResult[];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isAspectResult(value: unknown): value is AspectResult {
  return (
    isRecord(value) &&
    typeof value.aspect === "string" &&
    (value.sentiment === "Positive" ||
      value.sentiment === "Negative" ||
      value.sentiment === "Neutral") &&
    typeof value.score === "number" &&
    Number.isFinite(value.score)
  );
}

function isAnalysisResult(value: unknown): value is AnalysisResult {
  return (
    isRecord(value) &&
    typeof value.review === "string" &&
    (value.classification === "Genuine" ||
      value.classification === "Fake / Deceptive") &&
    (value.confidence === null ||
      (typeof value.confidence === "number" &&
        Number.isFinite(value.confidence) &&
        value.confidence >= 0 &&
        value.confidence <= 100)) &&
    Array.isArray(value.aspects) &&
    value.aspects.every(isAspectResult)
  );
}

const exampleReview =
  "The delivery was fast and the product quality was excellent, but the price was too high.";

export default function Home() {
  const [review, setReview] = useState("");
  const [rating, setRating] = useState("5");
  const [verifiedPurchase, setVerifiedPurchase] = useState("Y");
  const [isAnalysing, setIsAnalysing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");

  function loadExample() {
    setReview(exampleReview);
    setRating("4");
    setVerifiedPurchase("Y");
    setResult(null);
    setError("");
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (isAnalysing) return;
    if (!review.trim()) {
      setError("Please enter a review before analysing it.");
      return;
    }

    const selectedRating = Number(rating);
    if (!Number.isFinite(selectedRating) || selectedRating < 1 || selectedRating > 5) {
      setError("Select a product rating from 1 to 5 stars.");
      return;
    }
    if (verifiedPurchase !== "Y" && verifiedPurchase !== "N") {
      setError("Select whether the purchase was verified.");
      return;
    }

    setError("");
    setResult(null);
    setIsAnalysing(true);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          review_text: review.trim(),
          rating: selectedRating,
          verified_purchase: verifiedPurchase,
        }),
      });

      let data: unknown;
      try {
        data = await response.json();
      } catch {
        throw new Error(
          "The analysis service returned an unreadable response. Please try again.",
        );
      }

      if (!response.ok) {
        if (response.status === 400) {
          throw new Error(
            "The review could not be analysed. Check the review details and try again.",
          );
        }
        if (response.status >= 500) {
          throw new Error(
            "Unable to reach the analysis service. Please ensure it is running and try again.",
          );
        }
        throw new Error("The review could not be analysed. Please try again.");
      }

      if (!isAnalysisResult(data)) {
        throw new Error(
          "The analysis service returned an invalid result. Please try again.",
        );
      }

      setResult(data);
    } catch (caughtError) {
      if (caughtError instanceof TypeError) {
        setError(
          "Unable to connect to the analysis service. Please ensure it is running and try again.",
        );
      } else if (
        caughtError instanceof Error &&
        (caughtError.name === "TimeoutError" ||
          caughtError.name === "AbortError")
      ) {
        setError(
          "The analysis is taking longer than expected. Please try again.",
        );
      } else if (caughtError instanceof Error) {
        setError(caughtError.message);
      } else {
        setError("The review could not be analysed. Please try again.");
      }
    } finally {
      setIsAnalysing(false);
    }
  }

  function sentimentClasses(sentiment: Sentiment) {
    if (sentiment === "Positive") {
      return "border-emerald-200 bg-emerald-50 text-emerald-800";
    }
    if (sentiment === "Negative") {
      return "border-rose-200 bg-rose-50 text-rose-800";
    }
    return "border-slate-200 bg-slate-50 text-slate-700";
  }

  function classificationClasses(classification: Classification) {
    if (classification === "Genuine") {
      return "border-emerald-200 bg-emerald-50 text-emerald-800";
    }
    return "border-amber-200 bg-amber-50 text-amber-900";
  }

  return (
    <main className="short-viewport min-h-screen bg-[#f2f5f3] text-[#142d32]">
      <header className="border-b border-[#dce5e3] bg-white">
        <div className="app-header-inner mx-auto flex max-w-340 items-center justify-between gap-4 px-5 py-3 sm:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <div
              aria-hidden="true"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[#12383d] text-xs font-bold text-white shadow-sm"
            >
              FR
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-[#142d32]">
                Fake Review Detection
              </p>
              <p className="hidden text-xs text-[#63747a] sm:block">
                E-commerce review analysis system
              </p>
            </div>
          </div>
        </div>
      </header>

      <div className="page-content mx-auto max-w-340 px-5 pb-24 sm:px-8">
        <section className="hero-pattern relative flex flex-col justify-between gap-4 rounded-b-lg px-5 py-6 text-white sm:flex-row sm:items-end sm:px-8 sm:py-6">
          <div className="max-w-3xl">
            <p className="hero-eyebrow mb-3 text-xs font-semibold uppercase tracking-[0.14em] text-[#d7eaa3]">
              E-commerce review analysis
            </p>
            <h1 className="max-w-2xl text-3xl font-semibold leading-tight text-white sm:text-3xl">
              Review credibility, examined by aspect.
            </h1>
            <p className="hero-description mt-2 max-w-2xl text-sm leading-6 text-white/75 sm:text-base">
              Analyse e-commerce reviews using aspect-based sentiment analysis
              and machine learning.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs font-medium text-white/90 sm:max-w-62.5 sm:justify-end">
            <span className="rounded-md border border-white/20 bg-white/5 px-3 py-2">
              Aspect-based sentiment analysis
            </span>
            <span className="rounded-md border border-white/20 bg-white/5 px-3 py-2">
              SVM classification
            </span>
          </div>
        </section>

        <section
          aria-label="Review analysis workspace"
          className="workspace-grid grid items-stretch gap-4 py-4 lg:grid-cols-[minmax(0,1.08fr)_minmax(380px,0.92fr)] lg:gap-5 lg:py-4"
        >
          <section className="input-panel overflow-hidden rounded-lg border border-[#d9e2de] bg-white shadow-[0_8px_28px_rgba(20,45,50,0.06)]">
            <div className="input-panel-header flex items-start justify-between gap-4 border-b border-[#e5ecea] bg-[#fbfcfb] px-5 py-3 sm:px-5">
              <div>
                <h2 className="text-lg font-semibold text-[#142d32]">
                  Review input
                </h2>
                <p className="mt-1 text-sm text-[#63747a]">
                  Enter a product review and its purchase details.
                </p>
              </div>
              <button
                type="button"
                onClick={loadExample}
                disabled={isAnalysing}
                className="min-h-10 shrink-0 rounded-md border border-[#cbd9d6] px-3 text-sm font-semibold text-[#31545a] transition-colors hover:bg-[#edf4f1] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#16766d] disabled:cursor-not-allowed disabled:opacity-50"
              >
                Use example
              </button>
            </div>

            <form onSubmit={handleSubmit} className="review-form space-y-4 p-4 sm:p-5">
              <div>
                <div className="review-label-row mb-2 flex items-center justify-between gap-3">
                  <label
                    htmlFor="review"
                    className="text-sm font-semibold text-[#263c43]"
                  >
                    Review text <span className="text-[#a33d35]">*</span>
                  </label>
                  <div className="flex items-center gap-3 text-xs text-[#718187]">
                    <span className="hidden sm:inline">Enter to analyse · Shift+Enter for a new line</span>
                    <span className="tabular-nums">{review.length} characters</span>
                  </div>
                </div>
                <div className="review-editor-shell">
                  <textarea
                    id="review"
                    value={review}
                    onChange={(event) => {
                      setReview(event.target.value);
                      setResult(null);
                      setError("");
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        event.currentTarget.form?.requestSubmit();
                      }
                    }}
                    placeholder="Describe your experience with the product..."
                    rows={6}
                    required
                    disabled={isAnalysing}
                    className="h-40 w-full resize-none rounded-md bg-transparent px-4 py-3 text-sm leading-6 text-[#142d32] outline-none placeholder:text-[#89969a] disabled:cursor-not-allowed"
                  />
                </div>
              </div>

              <div className="metadata-grid grid gap-4 sm:grid-cols-2">
                <div>
                  <label
                    htmlFor="rating"
                    className="metadata-field-label mb-2 block text-sm font-semibold text-[#263c43]"
                  >
                    Product rating
                  </label>
                  <select
                    id="rating"
                    value={rating}
                    onChange={(event) => {
                      setRating(event.target.value);
                      setResult(null);
                      setError("");
                    }}
                    disabled={isAnalysing}
                    className="metadata-select min-h-10 w-full rounded-md border border-[#bdceca] bg-white px-3 text-sm text-[#142d32] outline-none focus-visible:border-[#16766d] focus-visible:ring-2 focus-visible:ring-[#16766d]/20"
                  >
                    <option value="1">1 star</option>
                    <option value="2">2 stars</option>
                    <option value="3">3 stars</option>
                    <option value="4">4 stars</option>
                    <option value="5">5 stars</option>
                  </select>
                </div>
                <div>
                  <label
                    htmlFor="verified"
                    className="metadata-field-label mb-2 block text-sm font-semibold text-[#263c43]"
                  >
                    Verified purchase
                  </label>
                  <select
                    id="verified"
                    value={verifiedPurchase}
                    onChange={(event) => {
                      setVerifiedPurchase(event.target.value);
                      setResult(null);
                      setError("");
                    }}
                    disabled={isAnalysing}
                    className="metadata-select min-h-10 w-full rounded-md border border-[#bdceca] bg-white px-3 text-sm text-[#142d32] outline-none focus-visible:border-[#16766d] focus-visible:ring-2 focus-visible:ring-[#16766d]/20"
                  >
                    <option value="Y">Yes</option>
                    <option value="N">No</option>
                  </select>
                </div>
              </div>

              {error && (
                <div
                  id="analysis-error"
                  role="alert"
                  className="rounded-r-md border-l-4 border-[#a33d35] bg-[#fff5f3] px-4 py-3 text-sm leading-5 text-[#7e302b]"
                >
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={isAnalysing}
                aria-describedby={error ? "analysis-error" : undefined}
                className="analyze-button flex min-h-11 w-full items-center justify-center gap-2 rounded-md bg-[#d7eaa3] px-5 text-sm font-bold text-[#17383d] shadow-sm transition-colors hover:bg-[#e2efbd] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#16766d] disabled:cursor-wait disabled:bg-[#cbd5cf]"
              >
                {isAnalysing ? (
                  <>
                    <span
                      aria-hidden="true"
                      className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white motion-reduce:animate-none"
                    />
                    Analysing review...
                  </>
                ) : (
                  "Analyse review"
                )}
              </button>
            </form>
          </section>

          <section
            aria-labelledby="results-heading"
            aria-live="polite"
            className="results-panel min-w-0 overflow-hidden rounded-lg border border-[#d9e2de] border-t-4 border-t-[#16766d] bg-white shadow-[0_8px_28px_rgba(20,45,50,0.06)]"
          >
            <div className="results-header border-b border-[#e5ecea] px-5 py-4 sm:px-6">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2
                    id="results-heading"
                    className="text-lg font-semibold text-[#142d32]"
                  >
                    Analysis results
                  </h2>
                  <p className="mt-1 text-sm text-[#63747a]">
                    Prediction and aspect-level sentiment
                  </p>
                </div>
                {result && (
                  <span className="border border-[#d0dfdc] px-2.5 py-1 text-xs font-medium text-[#53676d]">
                    {result.aspects.length} {result.aspects.length === 1 ? "aspect" : "aspects"}
                  </span>
                )}
              </div>
            </div>

            {isAnalysing ? (
              <div role="status" className="flex min-h-105 flex-col items-center justify-center px-6 text-center">
                <span
                  aria-hidden="true"
                  className="h-8 w-8 animate-spin rounded-full border-2 border-[#cbd9d6] border-t-[#28766d] motion-reduce:animate-none"
                />
                <p className="mt-4 text-sm font-semibold text-[#263c43]">
                  Analysing submitted review
                </p>
                <p className="mt-1 max-w-sm text-sm leading-6 text-[#63747a]">
                  Extracting aspects, evaluating sentiment and generating the prediction.
                </p>
              </div>
            ) : !result ? (
              <div className="flex min-h-105 flex-col justify-center bg-[#fbfcfb] px-6 py-10 sm:px-8">
                <span className="text-xs font-semibold uppercase tracking-[0.12em] text-[#28766d]">
                  Awaiting review input
                </span>
                <h3 className="mt-3 text-xl font-semibold text-[#263c43]">
                  Your analysis results will appear here.
                </h3>
                <p className="mt-3 max-w-md text-sm leading-6 text-[#63747a]">
                  The system identifies review aspects, determines their
                  sentiment and predicts whether the review is genuine or
                  deceptive.
                </p>
                <div className="mt-7 border-t border-[#e5ecea] pt-4 text-xs leading-5 text-[#718187]">
                  No prediction is displayed until a review has been submitted.
                </div>
              </div>
            ) : (
              <div className="result-body px-5 sm:px-6">
                <div className="prediction-summary border-b border-[#e5ecea] py-5">
                  <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
                    <div>
                      <p className="prediction-label text-xs font-semibold uppercase tracking-widest text-[#63747a]">
                        Prediction
                      </p>
                      <p
                        className={`prediction-value mt-1 inline-flex rounded-md border px-3 py-1.5 text-lg font-semibold ${classificationClasses(result.classification)}`}
                      >
                        {result.classification}
                      </p>
                    </div>
                    <div className="sm:text-right">
                      <p className="confidence-label text-xs font-semibold uppercase tracking-widest text-[#63747a]">
                        Prediction confidence
                      </p>
                      <p className="confidence-value mt-1 text-2xl font-semibold tabular-nums text-[#172a32]">
                        {result.confidence === null
                          ? "Unavailable"
                          : `${result.confidence.toFixed(2)}%`}
                      </p>
                    </div>
                  </div>
                  {result.confidence !== null && (
                    <div
                      className="mt-4 h-2 w-full bg-[#e7eeec]"
                      role="meter"
                      aria-label="Prediction confidence"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={result.confidence}
                    >
                      <div
                        className="h-full bg-[#16766d] transition-[width] duration-500 motion-reduce:transition-none"
                        style={{ width: `${result.confidence}%` }}
                      />
                    </div>
                  )}
                  <p className="confidence-note mt-2 text-xs text-[#718187]">
                    Confidence in this prediction; it is not a measure of model accuracy.
                  </p>
                </div>

                <div className="submitted-review border-b border-[#e5ecea] py-4">
                  <h3 className="text-xs font-semibold uppercase tracking-widest text-[#63747a]">
                    Submitted review
                  </h3>
                  <p className="mt-2 max-h-28 overflow-y-auto whitespace-pre-wrap wrap-break-word text-sm leading-6 text-[#40565d]">
                    {result.review}
                  </p>
                </div>

                <div className="aspect-results py-4">
                  <div className="flex items-center justify-between gap-3">
                    <h3 className="text-sm font-semibold text-[#263c43]">
                      Aspect-based sentiment
                    </h3>
                    <span className="text-xs text-[#718187]">
                      Aspect / sentiment / score
                    </span>
                  </div>

                  {result.aspects.length === 0 ? (
                    <p className="border-t border-[#e5ecea] py-5 text-sm leading-6 text-[#63747a]">
                      No recognised product aspects were identified in this review.
                    </p>
                  ) : (
                    <>
                      <div className="mt-3 hidden grid-cols-[minmax(0,1fr)_auto_72px] gap-4 border-y border-[#e5ecea] py-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#718187] sm:grid">
                        <span>Aspect</span>
                        <span>Sentiment</span>
                        <span className="text-right">Score</span>
                      </div>
                      <ul className="aspect-list divide-y divide-[#e5ecea]">
                        {result.aspects.map((item, index) => (
                          <li
                            key={`${item.aspect}-${index}`}
                            className="aspect-row grid grid-cols-[minmax(0,1fr)_auto_58px] items-center gap-2 py-3 sm:grid-cols-[minmax(0,1fr)_auto_72px] sm:gap-4"
                          >
                            <span className="wrap-break-word text-sm font-medium text-[#263c43]">
                              {item.aspect}
                            </span>
                            <span
                              className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${sentimentClasses(item.sentiment)}`}
                            >
                              {item.sentiment}
                            </span>
                            <span className="text-right text-sm font-semibold tabular-nums text-[#40565d]">
                              {item.score > 0 ? "+" : ""}
                              {item.score.toFixed(1)}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </>
                  )}
                </div>
              </div>
            )}
          </section>
        </section>

        <section
          aria-labelledby="pipeline-heading"
          className="rounded-lg border border-[#d9e2de] bg-white px-5 py-5 shadow-[0_4px_18px_rgba(20,45,50,0.04)] sm:px-6"
        >
          <div className="flex flex-col justify-between gap-1 sm:flex-row sm:items-end">
            <div>
              <h2
                id="pipeline-heading"
                className="text-sm font-semibold text-[#142d32]"
              >
                Analysis pipeline
              </h2>
              <p className="mt-1 text-xs text-[#718187]">
                Processing stages used to generate the displayed result
              </p>
            </div>
            <span className="text-xs text-[#718187]">ABSA + SVM</span>
          </div>
          <ol className="mt-5 grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-4 lg:grid-cols-7">
            {[
              ["01", "Review input"],
              ["02", "Text pre-processing"],
              ["03", "Aspect extraction"],
              ["04", "Aspect-level sentiment analysis"],
              ["05", "Feature processing: TF-IDF, ABSA and metadata"],
              ["06", "SVM classification"],
              ["07", "Prediction result"],
            ].map(([number, label]) => (
              <li key={number} className="min-w-0 rounded-md border border-[#e1e9e5] bg-[#f9fbf9] p-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-[#e3f0e9] text-[10px] font-bold tabular-nums text-[#16766d]">
                  {number}
                </span>
                <span className="mt-2 block text-xs font-semibold leading-5 text-[#40565d]">
                  {label}
                </span>
              </li>
            ))}
          </ol>
        </section>
      </div>

      <footer className="border-t border-[#dce5e3] bg-white">
        <div className="mx-auto flex max-w-340 flex-col gap-2 px-5 py-4 text-xs text-[#718187] sm:flex-row sm:items-center sm:justify-between sm:px-8">
          <p>Fake Review Detection using Aspect-Based Sentiment Analysis</p>
          <p>Aspect-based sentiment and SVM classification</p>
        </div>
      </footer>
    </main>
  );
}