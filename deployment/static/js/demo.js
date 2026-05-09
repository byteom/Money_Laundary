export function debounce(callback, waitMs) {
  let timerId = null;
  return (...args) => {
    if (timerId) {
      clearTimeout(timerId);
    }
    timerId = setTimeout(() => callback(...args), waitMs);
  };
}

export function bindSliderLabel(sliderId, valueId, formatter) {
  const slider = document.getElementById(sliderId);
  const value = document.getElementById(valueId);

  const update = () => {
    value.textContent = formatter(slider.value);
  };

  slider.addEventListener('input', update);
  update();
}
