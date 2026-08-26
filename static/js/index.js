const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const setupTableOfContents = () => {
    const tocLinks = [...document.querySelectorAll(".toc-link")];
    const sections = tocLinks
        .map((link) => document.querySelector(link.getAttribute("href")))
        .filter(Boolean);

    if (!sections.length) {
        return;
    }

    const setActiveSection = (id) => {
        tocLinks.forEach((link) => {
            const isActive = link.getAttribute("href") === `#${id}`;
            link.classList.toggle("is-active", isActive);
            if (isActive) {
                link.setAttribute("aria-current", "location");
            } else {
                link.removeAttribute("aria-current");
            }
        });
    };

    const observer = new IntersectionObserver(
        (entries) => {
            const visible = entries
                .filter((entry) => entry.isIntersecting)
                .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

            if (visible) {
                setActiveSection(visible.target.id);
            }
        },
        {
            rootMargin: "-28% 0px -58% 0px",
            threshold: [0.05, 0.2, 0.45],
        }
    );

    sections.forEach((section) => observer.observe(section));
};

const sequentialPath = "./static/videos/sequential_manipulation_480/";
const insertionPath = "./static/videos/fine_grained_insertion_480/";

const pairedTask = (label, human, robot, options = {}) => ({
    label,
    human,
    robot,
    description: "",
    tags: [],
    thumbnail: null,
    robotSpeed: null,
    ...options,
});

const demoGroups = {
    sequential: [
        pairedTask("Silver coffee cup, sponge, and gourd", `${sequentialPath}seq-01-human.mp4`, `${sequentialPath}seq-01-robot.mp4`, {
            description: "Place the silver coffee cup in the wooden basket. Place the sponge in the white tray. Place the gourd on the pink oval plate.",
        }),
        pairedTask("Glue stick, block, and bowl", `${sequentialPath}seq-02-human.mp4`, `${sequentialPath}seq-02-robot.mp4`, {
            description: "Place the glue stick on the brown-green plate. Place the yellow block in the wooden basket. Place the pink bowl in the white tray.",
        }),
        pairedTask("Bread, pear, and cup", `${sequentialPath}seq-03-human.mp4`, `${sequentialPath}seq-03-robot.mp4`, {
            description: "Place the pointed bread in the wooden basket. Place the green pear on the brown-green plate. Place the silver coffee cup in the round paper box.",
        }),
        pairedTask("Cup, pear, and sponge", `${sequentialPath}seq-04-human.mp4`, `${sequentialPath}seq-04-robot.mp4`, {
            description: "Place the wooden cup in the wooden basket. Place the green pear on the brown-green plate. Place the sponge in the white tray.",
        }),
    ],
    insertion: [
        pairedTask("Table leg inserting", `${insertionPath}insert-03-human.mp4`, `${insertionPath}insert-03-robot-2x.mp4?v=20260823-2x`, {
            description: "Move the white tabletop base to the left. Insert the green table leg into the bottom-right corner of the white tabletop, then insert the blue table leg into the top-left corner.",
            tags: ["Long-horizon", "Human-robot collaboration"],
            thumbnail: "./static/images/insertion/table-legs.webp",
            robotSpeed: "Speed ×2",
        }),
        pairedTask("Bulb inserting", `${insertionPath}insert-02-human.mp4`, `${insertionPath}insert-02-robot.mp4`, {
            description: "Insert the black bulb into the black base.",
            thumbnail: "./static/images/insertion/black-bulb.webp",
        }),
        pairedTask("Red column inserting", `${insertionPath}insert-01-human.mp4`, `${insertionPath}insert-01-robot.mp4`, {
            description: "Insert the red column into the black base.",
            thumbnail: "./static/images/insertion/red-column.webp",
        }),
    ],
};

const iclFilmPath = "./static/videos/human_video_prompt_gallery/";
const iclFilmTasks = [
    {
        label: "Which package contains the spicy peanuts?",
        category: 0,
        human: `${iclFilmPath}agibot-peanuts-human.mp4`,
        robot: `${iclFilmPath}agibot-peanuts-robot.mp4`,
    },
    {
        label: "Which items are the children's tableware?",
        category: 0,
        human: `${iclFilmPath}agibot-tableware-human.mp4`,
        robot: `${iclFilmPath}agibot-tableware-robot.mp4`,
    },
    {
        label: "Which bottle contains the coffee?",
        category: 0,
        human: `${iclFilmPath}agibot-coffee-human.mp4`,
        robot: `${iclFilmPath}agibot-coffee-robot.mp4`,
        speed: "1.5×",
    },
    {
        label: "How to zip up the coat?",
        category: 1,
        human: `${iclFilmPath}oxe-coat-human.mp4`,
        robot: `${iclFilmPath}oxe-coat-robot.mp4`,
        speed: "2×",
    },
    {
        label: "How to turn on the blue table lamp?",
        category: 1,
        human: `${iclFilmPath}robomind-lamp-human.mp4`,
        robot: `${iclFilmPath}robomind-lamp-robot.mp4`,
    },
    {
        label: "Place fries in a microwave and turn it on...",
        category: 2,
        human: `${iclFilmPath}robocoin-microwave-human.mp4`,
        robot: `${iclFilmPath}robocoin-microwave-robot.mp4`,
        speed: "6×",
    },
    {
        label: "Sort a cluttered desk...",
        category: 2,
        human: `${iclFilmPath}galaxea-desk-human.mp4`,
        robot: `${iclFilmPath}galaxea-desk-robot.mp4`,
        speed: "4×",
    },
    {
        label: "Which object is the leaf?",
        category: 0,
        human: `${iclFilmPath}galaxea-pick-human.mp4`,
        robot: `${iclFilmPath}galaxea-pick-robot.mp4`,
        speed: "3×",
    },
    {
        label: "How to place the block into the fixture?",
        category: 1,
        human: `${iclFilmPath}openloong-block-human.mp4`,
        robot: `${iclFilmPath}openloong-block-robot.mp4`,
        speed: "6×",
    },
    {
        label: "How to operate the oven door?",
        category: 1,
        human: `${iclFilmPath}robomind-oven-human.mp4`,
        robot: `${iclFilmPath}robomind-oven-robot.mp4`,
        speed: "4×",
    },
    {
        label: "Operate a smart toilet",
        category: 3,
        human: `${iclFilmPath}ruierman-toilet-human.mp4`,
        robot: `${iclFilmPath}ruierman-toilet-robot.mp4`,
        speed: "14×",
    },
    {
        label: "Weigh apples at checkout...",
        category: 2,
        human: `${iclFilmPath}ruierman-scale-human.mp4`,
        robot: `${iclFilmPath}ruierman-scale-robot.mp4`,
        speed: "5×",
    },
    {
        label: "Arrange files on desk...",
        category: 2,
        human: `${iclFilmPath}shv2-instruction-human.mp4`,
        robot: `${iclFilmPath}shv2-instruction-robot.mp4`,
        speed: "16×",
    },
    {
        label: "Pour from a teapot",
        category: 3,
        human: `${iclFilmPath}gallery-agibot-pour-human.mp4`,
        robot: `${iclFilmPath}gallery-agibot-pour-robot.mp4`,
    },
    {
        label: "Insert a marker",
        category: 3,
        human: `${iclFilmPath}gallery-agibot-marker-human.mp4`,
        robot: `${iclFilmPath}gallery-agibot-marker-robot.mp4`,
    },
    {
        label: "Make a sandwich",
        category: 3,
        human: `${iclFilmPath}gallery-interna1-sandwich-human.mp4`,
        robot: `${iclFilmPath}gallery-interna1-sandwich-robot.mp4`,
    },
    {
        label: "Fold short pants",
        category: 3,
        human: `${iclFilmPath}gallery-interna1-pants-human.mp4`,
        robot: `${iclFilmPath}gallery-interna1-pants-robot.mp4`,
    },
    {
        label: "Prepare a place setting",
        category: 3,
        human: `${iclFilmPath}gallery-galaxea-chef-human.mp4`,
        robot: `${iclFilmPath}gallery-galaxea-chef-robot.mp4`,
    },
    {
        label: "Rinse and place a plate",
        category: 3,
        human: `${iclFilmPath}gallery-oxe-rinse-human.mp4`,
        robot: `${iclFilmPath}gallery-oxe-rinse-robot.mp4`,
    },
    {
        label: "Manipulate a laptop",
        category: 3,
        human: `${iclFilmPath}gallery-robocoin-computer-human.mp4`,
        robot: `${iclFilmPath}gallery-robocoin-computer-robot.mp4`,
    },
    {
        label: "Pick up an umbrella",
        category: 3,
        human: `${iclFilmPath}gallery-robocoin-umbrella-human.mp4`,
        robot: `${iclFilmPath}gallery-robocoin-umbrella-robot.mp4`,
    },
    {
        label: "Cover a pot",
        category: 3,
        human: `${iclFilmPath}gallery-robomind-lid-human.mp4`,
        robot: `${iclFilmPath}gallery-robomind-lid-robot.mp4`,
    },
    {
        label: "Open a laptop",
        category: 3,
        human: `${iclFilmPath}gallery-robotwin-laptop-human.mp4`,
        robot: `${iclFilmPath}gallery-robotwin-laptop-robot.mp4`,
    },
    {
        label: "Operate a workstation",
        category: 3,
        human: `${iclFilmPath}gallery-ruierman-workstation-human.mp4`,
        robot: `${iclFilmPath}gallery-ruierman-workstation-robot.mp4`,
    },
];

const iclFilmStages = [
    {
        start: 0,
        category: 0,
        title: "Which object?",
        copy: "Identify the intended instance among visually similar objects.",
    },
    {
        start: 6000,
        category: 1,
        title: "How should it be manipulated?",
        copy: "Show the interaction required by an unfamiliar object.",
    },
    {
        start: 12000,
        category: 2,
        title: "What sequence is required?",
        copy: "Specify the order of actions and the state changes that complete the task.",
    },
    {
        start: 18000,
        category: null,
        title: "Human-robot ICL gallery.",
        copy: "",
    },
];

const setupIclFilm = () => {
    const film = document.querySelector("[data-icl-film]");
    if (!film) {
        return;
    }

    const gallery = film.querySelector("[data-film-gallery]");
    const chapterButtons = [...film.querySelectorAll("[data-film-stage]")];
    const title = film.querySelector("[data-film-title]");
    const copy = film.querySelector("[data-film-copy]");
    const toggle = film.querySelector("[data-film-toggle]");
    const toggleIcon = toggle.querySelector("i");
    const replay = film.querySelector("[data-film-replay]");
    const timeline = film.querySelector("[data-film-timeline]");
    const timelineFill = film.querySelector("[data-film-timeline-fill]");
    const time = film.querySelector("[data-film-time]");
    const duration = 24000;
    let elapsed = 0;
    let activeStage = -1;
    let animationFrame = null;
    let previousTimestamp = null;
    let isPlaying = false;
    let isVisible = false;
    let hasStarted = false;
    const categoryCounts = new Map();

    const makeVideo = (source, accessibleLabel, roleLabel, preload = "metadata") => {
        const wrapper = document.createElement("div");
        const video = document.createElement("video");
        const role = document.createElement("span");

        wrapper.className = "icl-pair-media";
        video.src = encodeURI(source);
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.preload = preload;
        video.defaultPlaybackRate = 1;
        video.playbackRate = 1;
        video.addEventListener("loadedmetadata", () => {
            video.playbackRate = 1;
        });
        video.setAttribute("aria-label", accessibleLabel);
        video.addEventListener("loadeddata", () => video.classList.add("is-ready"), { once: true });
        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
            video.classList.add("is-ready");
        }
        role.className = "icl-pair-role";
        role.textContent = roleLabel;
        wrapper.append(video, role);
        return wrapper;
    };

    const cards = iclFilmTasks.map((task, index) => {
        const card = document.createElement("article");
        const media = document.createElement("div");
        const heading = document.createElement("h4");
        const speed = document.createElement("span");

        card.className = "icl-pair-card";
        card.dataset.taskIndex = String(index);
        card.dataset.category = String(task.category);
        const categoryOrder = categoryCounts.get(task.category) || 0;
        categoryCounts.set(task.category, categoryOrder + 1);
        card.dataset.categoryOrder = String(categoryOrder);
        card.style.setProperty("--gallery-delay", `${Math.min(index, 18) * 21}ms`);
        media.className = "icl-pair-videos";
        heading.textContent = task.label;
        media.append(
            makeVideo(task.human, `${task.label}: Human demonstration`, "Human demonstration", task.category < 3 ? "auto" : "metadata"),
            makeVideo(task.robot, `${task.label}: Robot video`, "Robot Video", task.category < 3 ? "auto" : "metadata")
        );
        card.append(media, heading);

        if (task.speed) {
            speed.className = "icl-pair-speed";
            speed.textContent = `Robot ${task.speed}`;
            card.append(speed);
        }

        gallery.append(card);
        return card;
    });

    const videos = [...gallery.querySelectorAll("video")];
    let galleryPreloaded = false;

    const preloadGalleryVideos = () => {
        if (galleryPreloaded) {
            return;
        }
        galleryPreloaded = true;
        cards
            .filter((card) => Number(card.dataset.category) >= 3)
            .flatMap((card) => [...card.querySelectorAll("video")])
            .forEach((video) => {
                video.preload = "auto";
                video.load();
            });
    };

    const stageFromElapsed = () => {
        for (let index = iclFilmStages.length - 1; index >= 0; index -= 1) {
            if (elapsed >= iclFilmStages[index].start) {
                return index;
            }
        }
        return 0;
    };

    const syncVideoPlayback = () => {
        const stage = iclFilmStages[activeStage];
        videos.forEach((video) => {
            const card = video.closest(".icl-pair-card");
            const category = Number(card.dataset.category);
            const isVisibleTask = stage.category === null || category === stage.category;
            const shouldPlay = isVisible && isVisibleTask && (isPlaying || elapsed >= duration);
            video.playbackRate = stage.category === 2 && category === 2 && Number.isFinite(video.duration)
                ? Math.max(1, video.duration / 3.05)
                : 1;
            if (shouldPlay) {
                video.play().catch(() => {});
            } else {
                video.pause();
            }
        });
    };

    const updateToggle = () => {
        const atEnd = elapsed >= duration;
        toggleIcon.className = `fas ${isPlaying ? "fa-pause" : atEnd ? "fa-redo-alt" : "fa-play"}`;
        toggle.setAttribute("aria-label", isPlaying ? "Pause story" : atEnd ? "Replay story" : "Play story");
    };

    const updateProgress = () => {
        const seconds = Math.min(duration, Math.max(0, elapsed)) / 1000;
        const percent = (seconds / (duration / 1000)) * 100;
        timelineFill.style.width = `${percent}%`;
        timeline.setAttribute("aria-valuenow", seconds.toFixed(1));
        time.textContent = `0:${String(Math.floor(seconds)).padStart(2, "0")} / 0:24`;
    };

    const setStage = (index, animate = true) => {
        if (index === activeStage) {
            return;
        }

        const previousCards = cards.filter((card) => card.classList.contains("is-active"));
        const previousRects = new Map(previousCards.map((card) => [card, card.getBoundingClientRect()]));
        activeStage = index;
        const stage = iclFilmStages[index];
        const isGallery = stage.category === null;

        if (index >= 1) {
            preloadGalleryVideos();
        }

        film.classList.toggle("is-gallery", isGallery);
        film.dataset.stage = String(index);
        cards.forEach((card, cardIndex) => {
            card.classList.toggle("is-active", !isGallery && Number(card.dataset.category) === stage.category);
        });
        if (!isGallery) {
            cards
                .filter((card) => card.classList.contains("is-active"))
                .flatMap((card) => [...card.querySelectorAll("video")])
                .forEach((video) => {
                    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
                        video.currentTime = 0;
                    } else {
                        video.addEventListener("loadedmetadata", () => {
                            video.currentTime = 0;
                        }, { once: true });
                    }
                });
        }
        chapterButtons.forEach((button, buttonIndex) => {
            const isActive = buttonIndex === index;
            button.classList.toggle("is-active", isActive);
            button.setAttribute("aria-selected", String(isActive));
        });
        title.textContent = stage.title;
        copy.textContent = stage.copy;

        if (animate && !prefersReducedMotion && isGallery && previousCards.length) {
            previousCards.forEach((card, order) => {
                const previousRect = previousRects.get(card);
                const nextRect = card.getBoundingClientRect();
                const translateX = previousRect.left - nextRect.left;
                const translateY = previousRect.top - nextRect.top;
                const scaleX = previousRect.width / nextRect.width;
                const scaleY = previousRect.height / nextRect.height;
                card.classList.add("is-transition-source");
                const transition = card.animate(
                    [
                        { transformOrigin: "top left", transform: `translate(${translateX}px, ${translateY}px) scale(${scaleX}, ${scaleY})` },
                        { transformOrigin: "top left", transform: "none" },
                    ],
                    { duration: 450 + order * 35, easing: "cubic-bezier(0.16, 1, 0.3, 1)" }
                );
                transition.finished.finally(() => card.classList.remove("is-transition-source"));
            });
        }

        syncVideoPlayback();
    };

    const stop = () => {
        isPlaying = false;
        previousTimestamp = null;
        if (animationFrame) {
            window.cancelAnimationFrame(animationFrame);
            animationFrame = null;
        }
        updateToggle();
        syncVideoPlayback();
    };

    const tick = (timestamp) => {
        if (!isPlaying) {
            return;
        }
        if (previousTimestamp !== null) {
            elapsed = Math.min(duration, elapsed + (timestamp - previousTimestamp));
        }
        previousTimestamp = timestamp;
        setStage(stageFromElapsed());
        updateProgress();

        if (elapsed >= duration) {
            stop();
            return;
        }
        animationFrame = window.requestAnimationFrame(tick);
    };

    const start = (restart = false) => {
        if (prefersReducedMotion || !isVisible || document.hidden) {
            return;
        }
        if (restart || elapsed >= duration) {
            elapsed = 0;
            setStage(0, false);
            updateProgress();
        }
        if (isPlaying) {
            return;
        }
        isPlaying = true;
        previousTimestamp = null;
        updateToggle();
        syncVideoPlayback();
        animationFrame = window.requestAnimationFrame(tick);
    };

    const seekTo = (nextElapsed, shouldResume = isPlaying) => {
        elapsed = Math.min(duration, Math.max(0, nextElapsed));
        setStage(stageFromElapsed());
        updateProgress();
        if (shouldResume && elapsed < duration) {
            start();
        } else {
            stop();
        }
    };

    chapterButtons.forEach((button, index) => {
        button.addEventListener("click", () => {
            isVisible = true;
            seekTo(iclFilmStages[index].start, true);
        });
    });

    toggle.addEventListener("click", () => {
        isVisible = true;
        if (isPlaying) {
            stop();
        } else {
            start(elapsed >= duration);
        }
    });
    replay.addEventListener("click", () => {
        isVisible = true;
        start(true);
    });
    timeline.addEventListener("pointerdown", (event) => {
        isVisible = true;
        const rect = timeline.getBoundingClientRect();
        seekTo(((event.clientX - rect.left) / rect.width) * duration, isPlaying);
    });
    timeline.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
            return;
        }
        event.preventDefault();
        if (event.key === "Home") {
            seekTo(0, false);
        } else if (event.key === "End") {
            seekTo(duration, false);
        } else {
            seekTo(elapsed + (event.key === "ArrowRight" ? 1000 : -1000), isPlaying);
        }
    });

    if (prefersReducedMotion) {
        elapsed = duration;
        setStage(3, false);
        updateProgress();
        stop();
        film.classList.add("is-reduced-motion");
        return;
    }

    setStage(0, false);
    updateProgress();
    const observer = new IntersectionObserver((entries) => {
        isVisible = entries.some((entry) => entry.isIntersecting);
        if (isVisible && !hasStarted) {
            hasStarted = true;
            start();
        } else if (!isVisible) {
            stop();
        }
    }, { threshold: 0.35 });

    observer.observe(film);
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            stop();
        }
    });
};

const formatVideoTime = (seconds) => {
    if (!Number.isFinite(seconds) || seconds < 0) {
        return "0:00";
    }

    const totalSeconds = Math.floor(seconds);
    const minutes = Math.floor(totalSeconds / 60);
    const remainingSeconds = String(totalSeconds % 60).padStart(2, "0");
    return `${minutes}:${remainingSeconds}`;
};

const bindVideoProgress = (video) => {
    const shell = video.closest(".video-shell");
    const playToggle = shell.querySelector(".video-play-toggle");
    const progress = shell.querySelector(".video-progress");
    const fill = shell.querySelector(".video-progress-fill");
    const currentTime = shell.querySelector(".video-time-current");
    const durationTime = shell.querySelector(".video-time-duration");
    let isDragging = false;
    let pendingSeekRatio = null;

    const progressFromEvent = (event) => {
        const rect = progress.getBoundingClientRect();
        const x = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
        return rect.width ? x / rect.width : 0;
    };

    const updateProgress = () => {
        const duration = video.duration;
        const ratio = Number.isFinite(duration) && duration > 0 ? video.currentTime / duration : 0;
        const percent = Math.min(Math.max(ratio * 100, 0), 100);
        fill.style.width = `${percent}%`;
        progress.setAttribute("aria-valuenow", String(Math.round(percent)));
        currentTime.textContent = formatVideoTime(video.currentTime);
        durationTime.textContent = formatVideoTime(duration);
    };

    const updatePlaybackState = () => {
        const isPlaying = !video.paused && !video.ended;
        playToggle.classList.toggle("is-playing", isPlaying);
        playToggle.setAttribute("aria-label", isPlaying ? "Pause video" : "Play video");
    };

    const seekToRatio = (ratio) => {
        const clampedRatio = Math.min(Math.max(ratio, 0), 1);

        if (!Number.isFinite(video.duration) || video.duration <= 0) {
            pendingSeekRatio = clampedRatio;
            return;
        }

        video.currentTime = clampedRatio * video.duration;
        updateProgress();
    };

    progress.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        isDragging = true;
        progress.classList.add("is-dragging");
        progress.setPointerCapture(event.pointerId);
        seekToRatio(progressFromEvent(event));
    });

    progress.addEventListener("pointermove", (event) => {
        if (isDragging) {
            seekToRatio(progressFromEvent(event));
        }
    });

    const finishDrag = (event) => {
        isDragging = false;
        progress.classList.remove("is-dragging");
        if (event && progress.hasPointerCapture(event.pointerId)) {
            progress.releasePointerCapture(event.pointerId);
        }
    };

    progress.addEventListener("pointerup", finishDrag);
    progress.addEventListener("pointercancel", finishDrag);

    progress.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
            return;
        }

        event.preventDefault();
        const step = event.key === "ArrowRight" ? 0.05 : -0.05;
        const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1;
        seekToRatio((video.currentTime / duration) + step);
    });

    playToggle.addEventListener("click", () => {
        if (video.paused || video.ended) {
            video.play().catch(() => {});
        } else {
            video.pause();
        }
    });

    video.addEventListener("timeupdate", updateProgress);
    video.addEventListener("loadedmetadata", () => {
        if (pendingSeekRatio !== null) {
            const ratio = pendingSeekRatio;
            pendingSeekRatio = null;
            seekToRatio(ratio);
        } else {
            updateProgress();
        }
    });
    video.addEventListener("durationchange", updateProgress);
    video.addEventListener("play", updatePlaybackState);
    video.addEventListener("pause", updatePlaybackState);
    video.addEventListener("ended", updatePlaybackState);
    updatePlaybackState();
};

const setupRobotwinVideoControls = () => {
    document.querySelectorAll(".robotwin-task video").forEach((video) => {
        if (video.closest(".video-shell")) {
            return;
        }

        const shell = document.createElement("div");
        const controls = document.createElement("div");
        const playToggle = document.createElement("button");
        const currentTime = document.createElement("span");
        const progress = document.createElement("div");
        const progressFill = document.createElement("div");
        const durationTime = document.createElement("span");
        const videoLabel = video.getAttribute("aria-label") || "RoboTwin evaluation video";

        shell.className = "video-shell robotwin-video-shell";
        controls.className = "video-controls";
        playToggle.className = "video-play-toggle is-playing";
        playToggle.type = "button";
        playToggle.setAttribute("aria-label", "Pause video");
        currentTime.className = "video-time video-time-current";
        currentTime.textContent = "0:00";
        progress.className = "video-progress";
        progress.setAttribute("role", "slider");
        progress.tabIndex = 0;
        progress.setAttribute("aria-label", `${videoLabel} progress`);
        progress.setAttribute("aria-valuemin", "0");
        progress.setAttribute("aria-valuemax", "100");
        progress.setAttribute("aria-valuenow", "0");
        progressFill.className = "video-progress-fill";
        durationTime.className = "video-time video-time-duration";
        durationTime.textContent = "0:00";

        progress.append(progressFill);
        controls.append(playToggle, currentTime, progress, durationTime);
        video.before(shell);
        shell.append(video, controls);
        bindVideoProgress(video);
    });
};

const warmedVideoSources = new Map();

const warmVideoSource = (source) => {
    const encodedSource = encodeURI(source);
    if (warmedVideoSources.has(encodedSource)) {
        return warmedVideoSources.get(encodedSource);
    }

    const video = document.createElement("video");
    video.preload = "auto";
    video.muted = true;
    video.playsInline = true;
    video.src = encodedSource;
    video.load();
    warmedVideoSources.set(encodedSource, video);
    return video;
};

const renderDemo = (block) => {
    const tasks = demoGroups[block.dataset.demo];
    const taskStrip = block.querySelector(".task-strip");
    const humanVideo = block.querySelector(".human-video");
    const robotVideo = block.querySelector(".robot-video");
    const taskDetail = block.querySelector("[data-task-detail]");
    const activeTaskName = taskDetail?.querySelector(".active-task-name");
    const activeTaskDescription = taskDetail?.querySelector(".active-task-description");
    const activeTaskTags = taskDetail?.querySelector(".active-task-tags");
    const robotSpeedBadge = block.querySelector("[data-video-speed]");
    const scrollLeft = block.querySelector(".task-scroll-left");
    const scrollRight = block.querySelector(".task-scroll-right");
    let activeIndex = 0;

    if (!tasks) {
        return;
    }

    const warmTask = (index) => {
        const task = tasks[(index + tasks.length) % tasks.length];
        warmVideoSource(task.human);
        warmVideoSource(task.robot);
    };

    const warmInactiveTasks = () => {
        tasks.forEach((_, index) => {
            if (index !== activeIndex) {
                warmTask(index);
            }
        });
    };

    const setActive = (index, options = {}) => {
        activeIndex = (index + tasks.length) % tasks.length;
        const task = tasks[activeIndex];

        buttons.forEach((button, buttonIndex) => {
            const isActive = buttonIndex === activeIndex;
            button.classList.toggle("is-active", isActive);
            button.setAttribute("aria-selected", String(isActive));
            button.tabIndex = isActive ? 0 : -1;
        });

        humanVideo.src = encodeURI(task.human);
        robotVideo.src = encodeURI(task.robot);
        humanVideo.setAttribute("aria-label", `${task.label} human video`);
        robotVideo.setAttribute("aria-label", `${task.label} robot video`);

        if (robotSpeedBadge) {
            robotSpeedBadge.hidden = !task.robotSpeed;
            robotSpeedBadge.textContent = task.robotSpeed || "";
        }

        if (taskDetail) {
            activeTaskName.textContent = task.label;
            activeTaskDescription.textContent = task.description;
            activeTaskTags.replaceChildren(...task.tags.map((tag) => {
                const tagElement = document.createElement("span");
                tagElement.textContent = tag;
                return tagElement;
            }));
        }

        humanVideo.play().catch(() => {});
        robotVideo.play().catch(() => {});

        if (options.scroll !== false) {
            buttons[activeIndex].scrollIntoView({
                behavior: prefersReducedMotion ? "auto" : "smooth",
                inline: "center",
                block: "nearest",
            });
        }
    };

    const buttons = tasks.map((task, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "task-button";
        button.setAttribute("role", "tab");
        if (task.thumbnail) {
            button.classList.add("task-button--visual");
            const thumbnail = document.createElement("img");
            const label = document.createElement("span");
            thumbnail.src = task.thumbnail;
            thumbnail.alt = "";
            thumbnail.loading = "lazy";
            thumbnail.width = 320;
            thumbnail.height = 180;
            label.textContent = task.label;
            button.append(thumbnail, label);
        } else {
            button.textContent = task.label;
        }
        button.addEventListener("pointerenter", () => warmTask(index), { once: true });
        button.addEventListener("focus", () => warmTask(index), { once: true });
        button.addEventListener("pointerdown", () => warmTask(index), { once: true });
        button.addEventListener("click", () => setActive(index));
        button.addEventListener("keydown", (event) => {
            if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
                event.preventDefault();
                const offset = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1;
                setActive(activeIndex + offset);
                buttons[(activeIndex + tasks.length) % tasks.length].focus();
            }
        });
        taskStrip.appendChild(button);
        return button;
    });

    scrollLeft?.addEventListener("click", () => setActive(activeIndex - 1));
    scrollRight?.addEventListener("click", () => setActive(activeIndex + 1));
    [humanVideo, robotVideo].forEach(bindVideoProgress);

    setActive(0, { scroll: false });

    const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
    if (!connection?.saveData) {
        const observer = new IntersectionObserver((entries) => {
            if (!entries.some((entry) => entry.isIntersecting)) {
                return;
            }

            observer.disconnect();
            if ("requestIdleCallback" in window) {
                window.requestIdleCallback(warmInactiveTasks, { timeout: 1000 });
            } else {
                window.setTimeout(warmInactiveTasks, 0);
            }
        }, { rootMargin: "1000px 0px", threshold: 0 });

        observer.observe(block);
    }
};

const setupDataCompositionVideos = () => {
    const composition = document.querySelector("[data-data-composition]");
    if (!composition) {
        return;
    }

    const videos = [...composition.querySelectorAll("[data-composition-video]")];
    let isVisible = false;

    videos.forEach((video) => {
        const offset = Number(video.dataset.offset || 0);
        if (!offset) {
            return;
        }

        video.addEventListener("loadedmetadata", () => {
            if (Number.isFinite(video.duration) && video.duration > 0) {
                video.currentTime = Math.min(offset, Math.max(video.duration - 0.1, 0));
            }
        }, { once: true });
    });

    const setPlayback = (shouldPlay) => {
        videos.forEach((video) => {
            if (!shouldPlay) {
                video.pause();
                return;
            }

            const playback = video.play();
            if (playback) {
                playback.catch(() => {});
            }
        });
    };

    if (prefersReducedMotion) {
        setPlayback(false);
        return;
    }

    const observer = new IntersectionObserver((entries) => {
        isVisible = entries.some((entry) => entry.isIntersecting);
        setPlayback(isVisible && !document.hidden);
    }, { threshold: 0.08 });

    observer.observe(composition);
    document.addEventListener("visibilitychange", () => {
        setPlayback(isVisible && !document.hidden);
    });
};

const setupFrameworkFigureVideos = () => {
    const figure = document.querySelector("[data-framework-figure]");
    if (!figure) {
        return;
    }

    const videos = [...figure.querySelectorAll("[data-framework-video]")];
    let isVisible = false;

    videos.forEach((video) => {
        const revealVideo = () => video.classList.add("is-ready");
        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
            revealVideo();
        } else {
            video.addEventListener("loadeddata", revealVideo, { once: true });
        }

        const offset = Number(video.dataset.offset || 0);
        const configurePlayback = () => {
            if (offset && Number.isFinite(video.duration) && video.duration > 0) {
                video.currentTime = Math.min(offset, Math.max(video.duration - 0.1, 0));
            }

            const fitDuration = Number(video.dataset.fitDuration || 0);
            if (fitDuration && Number.isFinite(video.duration) && video.duration > fitDuration) {
                const playbackRate = video.duration / fitDuration;
                video.defaultPlaybackRate = playbackRate;
                video.playbackRate = playbackRate;
            }
        };

        if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
            configurePlayback();
        } else {
            video.addEventListener("loadedmetadata", configurePlayback, { once: true });
        }
    });

    const setPlayback = (shouldPlay) => {
        videos.forEach((video) => {
            if (!shouldPlay) {
                video.pause();
                return;
            }

            const playback = video.play();
            if (playback) {
                playback.catch(() => {});
            }
        });
    };

    if (prefersReducedMotion) {
        setPlayback(false);
        return;
    }

    const observer = new IntersectionObserver((entries) => {
        isVisible = entries.some((entry) => entry.isIntersecting);
        setPlayback(isVisible && !document.hidden);
    }, { rootMargin: "12% 0px", threshold: 0.08 });

    observer.observe(figure);
    document.addEventListener("visibilitychange", () => {
        setPlayback(isVisible && !document.hidden);
    });
};

const setupPipelineAnimation = () => {
    const pipeline = document.querySelector("[data-pipeline]");
    if (!pipeline) {
        return;
    }

    const steps = [...pipeline.querySelectorAll("[data-pipeline-step]")]
        .sort((a, b) => Number(a.dataset.pipelineStep) - Number(b.dataset.pipelineStep));
    const links = [...pipeline.querySelectorAll("[data-pipeline-link]")];
    const videos = [...pipeline.querySelectorAll("[data-pipeline-video]")];
    let activeIndex = 0;
    let timerId = null;
    let isVisible = false;

    const paint = () => {
        steps.forEach((step, index) => step.classList.toggle("is-active", index === activeIndex));
        links.forEach((link) => {
            link.classList.toggle("is-active", Number(link.dataset.pipelineLink) === activeIndex);
        });
    };

    const setVideoPlayback = (shouldPlay) => {
        videos.forEach((video) => {
            if (!shouldPlay) {
                video.pause();
                return;
            }

            const playback = video.play();
            if (playback) {
                playback.catch(() => {});
            }
        });
    };

    const stop = () => {
        window.clearInterval(timerId);
        timerId = null;
        pipeline.classList.remove("is-running");
        pipeline.classList.add("is-paused");
        setVideoPlayback(false);
    };

    const start = () => {
        if (timerId || prefersReducedMotion || !isVisible || document.hidden) {
            return;
        }

        pipeline.classList.add("is-running");
        pipeline.classList.remove("is-paused");
        setVideoPlayback(true);
        paint();
        timerId = window.setInterval(() => {
            activeIndex = (activeIndex + 1) % steps.length;
            paint();
        }, 1100);
    };

    if (prefersReducedMotion) {
        pipeline.classList.add("is-reduced-motion");
        setVideoPlayback(false);
        return;
    }

    const observer = new IntersectionObserver((entries) => {
        isVisible = entries.some((entry) => entry.isIntersecting);
        if (isVisible) {
            start();
        } else {
            stop();
        }
    }, { threshold: 0.15 });

    observer.observe(pipeline);
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            stop();
        } else {
            start();
        }
    });
};

const setupChartAnimation = () => {
    const chart = document.querySelector(".results-chart");
    if (!chart) {
        return;
    }

    if (prefersReducedMotion) {
        chart.classList.add("is-visible");
        return;
    }

    const observer = new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
            chart.classList.add("is-visible");
            observer.disconnect();
        }
    }, { threshold: 0.25 });

    observer.observe(chart);
};

setupTableOfContents();
setupIclFilm();
document.querySelectorAll(".demo-block").forEach(renderDemo);
setupRobotwinVideoControls();
setupPipelineAnimation();
setupDataCompositionVideos();
setupFrameworkFigureVideos();
setupChartAnimation();
