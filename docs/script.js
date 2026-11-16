/*
  CMSC 170 Research Website

  JavaScript is only used for two small features:
  1. Drawing the dots in our performance plots.
  2. Revealing sections as the user scrolls.

  All experiment values are written in index.html so they are easy to check
*/

// Creates an SVG element 
function createSvgElement(tagName, attributes) {
  const svgElement = document.createElementNS("http://www.w3.org/2000/svg", tagName);

  for (const [name, value] of Object.entries(attributes)) {
    svgElement.setAttribute(name, value);
  }

  return svgElement;
}

// Draw one dot on a scale that starts at 90% and ends at 100%
function drawDot(track) {
  const score = Number(track.dataset.score);
  const width = 710;
  const height = 37;

  // Convert a score to its horizontal position in the SVG.
  // Example: 95% is halfway between 90% and 100%.
  const x = ((score - 90) / 10) * width;

  const svg = createSvgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "none",
    "aria-hidden": "true" // The score is already written beside the plot.
  });

  // A thin horizontal line through the dot plot.
  svg.appendChild(createSvgElement("line", {
    x1: 0, y1: 18, x2: width, y2: 18,
    stroke: "#c7cfc7", "stroke-width": 1
  }));

  // Grid marks at 90%, 92%, 94%, 96%, 98%, and 100%.
  for (let percent = 90; percent <= 100; percent += 2) {
    const tickX = ((percent - 90) / 10) * width;
    svg.appendChild(createSvgElement("line", {
      x1: tickX, y1: 9, x2: tickX, y2: 27,
      stroke: "#b7c5bb", "stroke-width": 1
    }));
  }

  // The marker showing the model's actual score.
  svg.appendChild(createSvgElement("circle", {
    cx: x, cy: 18, r: 7,
    fill: "#234740"
  }));

  track.appendChild(svg);
}

// Draw all six markers: three accuracy values and three recall values.
document.querySelectorAll(".dot-track").forEach(drawDot);

// Fade each section into view once, only when motion is allowed.
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

if ("IntersectionObserver" in window && !reducedMotion) {
  document.documentElement.classList.add("js");

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target); // No repeat animation.
      }
    });
  }, { threshold: 0.08 });

  document.querySelectorAll(".reveal").forEach((section) => {
    observer.observe(section);
  });
}
