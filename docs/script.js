/* CMSC 170 Research Website interactions */

function createSvgElement(tagName, attributes) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tagName);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  return element;
}

function drawDot(track) {
  const score = Number(track.dataset.score);
  if (!Number.isFinite(score)) return;
  const width = 710;
  const height = 37;
  const x = ((score - 90) / 10) * width;
  const svg = createSvgElement("svg", { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "none", "aria-hidden": "true" });
  svg.appendChild(createSvgElement("line", { x1: 0, y1: 18, x2: width, y2: 18, stroke: "#d5ddd5", "stroke-width": 1 }));
  for (let percent = 90; percent <= 100; percent += 2) {
    const tickX = ((percent - 90) / 10) * width;
    svg.appendChild(createSvgElement("line", { x1: tickX, y1: 9, x2: tickX, y2: 27, stroke: "#c5d1c7", "stroke-width": 1 }));
  }
  svg.appendChild(createSvgElement("circle", { cx: x, cy: 18, r: 7, fill: score >= 97 ? "#1e5145" : "#9aab9e" }));
  track.replaceChildren(svg);
}

document.querySelectorAll(".dot-track").forEach(drawDot);

// Mobile navigation
const menuToggle = document.querySelector(".menu-toggle");
const navigation = document.querySelector(".nav-links");
if (menuToggle && navigation) {
  menuToggle.addEventListener("click", () => {
    const expanded = menuToggle.getAttribute("aria-expanded") === "true";
    menuToggle.setAttribute("aria-expanded", String(!expanded));
    menuToggle.setAttribute("aria-label", expanded ? "Open navigation" : "Close navigation");
    navigation.classList.toggle("open", !expanded);
  });
  navigation.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
    menuToggle.setAttribute("aria-expanded", "false");
    menuToggle.setAttribute("aria-label", "Open navigation");
    navigation.classList.remove("open");
  }));
}

// Reading progress indicator
const progressBar = document.getElementById("scroll-progress-bar");
let scrollQueued = false;
function updateProgress() {
  if (!progressBar) return;
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const progress = scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0;
  progressBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
  scrollQueued = false;
}
window.addEventListener("scroll", () => {
  if (!scrollQueued) {
    window.requestAnimationFrame(updateProgress);
    scrollQueued = true;
  }
}, { passive: true });
updateProgress();

// Reveal sections as they enter the viewport, respecting reduced-motion settings.
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
if ("IntersectionObserver" in window && !reducedMotion) {
  document.documentElement.classList.add("js");
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08, rootMargin: "0px 0px -35px 0px" });
  document.querySelectorAll(".reveal").forEach((section) => observer.observe(section));
}

// Highlight the navigation item for the section currently in view.
const navLinks = [...document.querySelectorAll('.nav-links a[href^="#"]')];
const observedSections = navLinks.map((link) => document.querySelector(link.getAttribute("href"))).filter(Boolean);
if ("IntersectionObserver" in window && observedSections.length) {
  const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      navLinks.forEach((link) => {
        const active = link.getAttribute("href") === `#${entry.target.id}`;
        link.classList.toggle("active", active);
        if (active) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    });
  }, { rootMargin: "-25% 0px -65% 0px" });
  observedSections.forEach((section) => sectionObserver.observe(section));
}

// Accessible interactive model explainer tabs.
const explorerTabs = [...document.querySelectorAll('.explorer-tab')];
const explorerPanels = [...document.querySelectorAll('.explorer-panel')];
function activateExplorerTab(tab, moveFocus = false) {
  const model = tab.dataset.model;
  explorerTabs.forEach((item) => {
    const selected = item === tab;
    item.classList.toggle('active', selected);
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
  });
  explorerPanels.forEach((panel) => { panel.hidden = panel.id !== `panel-${model}`; });
  if (moveFocus) tab.focus();
}
explorerTabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateExplorerTab(tab));
  tab.addEventListener('keydown', (event) => {
    let next = index;
    if (event.key === 'ArrowRight') next = (index + 1) % explorerTabs.length;
    else if (event.key === 'ArrowLeft') next = (index - 1 + explorerTabs.length) % explorerTabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = explorerTabs.length - 1;
    else return;
    event.preventDefault();
    activateExplorerTab(explorerTabs[next], true);
  });
});
