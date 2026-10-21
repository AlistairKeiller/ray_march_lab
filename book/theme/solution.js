// Label the collapsed solution blocks. The markdown leaves <summary> empty so the
// text can live here, and the pages don't have to repeat it.
document.querySelectorAll(".solution-details").forEach((details) => {
  const summary = details.querySelector("summary");
  const label = () => {
    summary.textContent = details.open ? "Hide Solution" : "Reveal Solution";
  };
  label();
  details.addEventListener("toggle", label);
});
