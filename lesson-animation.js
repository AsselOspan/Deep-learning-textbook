(() => {
  'use strict';

  function markup(id, escape) {
    const lesson = window.DL_ANIMATIONS?.[id];
    if (!lesson?.frames.length) return '';
    return [
      '<section id="lesson-animation" class="lesson-animation" aria-labelledby="lesson-animation-title">',
      '<p class="eyebrow">Анимация к главе</p>',
      '<h2 id="lesson-animation-title">' + escape(lesson.title) + '</h2>',
      '<div class="animation-nodes" data-animation-nodes></div>',
      '<p class="animation-caption" data-animation-caption aria-live="polite"></p>',
      '<div class="animation-timeline">',
      '<label><span>Шаг анимации</span><input type="range" min="1" step="1" max="' + lesson.frames.length + '" value="1"></label>',
      '<span data-animation-progress></span></div>',
      '<div class="animation-controls">',
      '<button class="button primary" type="button" data-action="play">Воспроизвести</button>',
      '<button class="button" type="button" data-action="previous">Предыдущий шаг</button>',
      '<button class="button" type="button" data-action="next">Следующий шаг</button>',
      '</div></section>'
    ].join('');
  }

  function mount(section, id) {
    const { frames } = window.DL_ANIMATIONS[id];
    const nodes = section.querySelector('[data-animation-nodes]');
    const caption = section.querySelector('[data-animation-caption]');
    const slider = section.querySelector('input[type=range]');
    const progress = section.querySelector('[data-animation-progress]');
    const play = section.querySelector('[data-action=play]');
    const previous = section.querySelector('[data-action=previous]');
    const next = section.querySelector('[data-action=next]');
    let index = 0;
    let timer;

    function stop() {
      clearInterval(timer);
      timer = undefined;
      play.textContent = index === frames.length - 1 ? 'Повторить' : 'Воспроизвести';
    }

    function render() {
      const frame = frames[index];
      nodes.replaceChildren(...frame.nodes.map((node, position) => {
        const card = document.createElement('div');
        card.className = 'animation-node';
        card.classList.toggle('current', position === frame.active);
        const label = document.createElement('span');
        label.textContent = node.label;
        const value = document.createElement('strong');
        value.textContent = node.value;
        card.append(label, value);
        if (typeof node.level === 'number') {
          const bar = document.createElement('span');
          bar.className = 'animation-bar';
          bar.setAttribute('aria-hidden', 'true');
          bar.style.setProperty('--level', (node.level * 100) + '%');
          card.append(bar);
        }
        return card;
      }));
      caption.textContent = frame.caption;
      slider.value = String(index + 1);
      slider.setAttribute('aria-valuetext', 'Шаг ' + (index + 1) + ' из ' + frames.length);
      progress.textContent = (index + 1) + ' / ' + frames.length;
      previous.disabled = index === 0;
      next.disabled = index === frames.length - 1;
    }

    function seek(step) {
      index = Math.max(0, Math.min(frames.length - 1, step));
      render();
      stop();
    }

    previous.addEventListener('click', () => seek(index - 1));
    next.addEventListener('click', () => seek(index + 1));
    slider.addEventListener('input', () => seek(Number(slider.value) - 1));
    play.addEventListener('click', () => {
      if (timer !== undefined) {
        stop();
        return;
      }
      if (index === frames.length - 1) {
        index = 0;
        render();
      }
      play.textContent = 'Пауза';
      timer = setInterval(() => {
        index += 1;
        render();
        if (index === frames.length - 1) stop();
      }, 3800);
    });
    render();
    return stop;
  }

  window.DLLessonAnimation = { markup, mount };
})();

