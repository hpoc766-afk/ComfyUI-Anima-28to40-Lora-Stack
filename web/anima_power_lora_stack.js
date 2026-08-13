import { app } from "../../scripts/app.js";

const NODE_CLASS = "Anima28To40PowerLoraStack";
const ROW_PREFIX = "lora_";
const DEFAULT_WIDTH = 430;
const SEARCH_STYLE_ID = "anima-lora-search-style";
const SEARCH_RESULT_LIMIT = 250;
let activeLoraChooser = null;

function canvasScale() {
    const scale = Number(app.canvas?.ds?.scale);
    return Number.isFinite(scale) && scale > 0 ? scale : 1;
}

function canvasLineWidth() {
    return 1 / canvasScale();
}

function isLowQuality() {
    return canvasScale() < 0.5;
}

function rowWidgets(node) {
    return (node.widgets || []).filter((widget) => widget.name?.startsWith(ROW_PREFIX));
}

function markChanged(node) {
    node.graph?.setDirtyCanvas(true, true);
    app.graph?.setDirtyCanvas(true, true);
}

function nextRowName(node) {
    let maxIndex = 0;
    for (const widget of rowWidgets(node)) {
        const index = Number.parseInt(widget.name.slice(ROW_PREFIX.length), 10);
        if (Number.isFinite(index)) {
            maxIndex = Math.max(maxIndex, index);
        }
    }
    return `${ROW_PREFIX}${maxIndex + 1}`;
}

function renumberRows(node) {
    rowWidgets(node).forEach((widget, index) => {
        widget.name = `${ROW_PREFIX}${index + 1}`;
    });
}

function fitText(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) {
        return text;
    }
    let output = text;
    while (output.length > 3 && ctx.measureText(`${output}…`).width > maxWidth) {
        output = output.slice(0, -1);
    }
    return `${output}…`;
}

function columnLayout(width) {
    const margin = 10;
    const gap = 6;
    const toggleWidth = 34;
    const strengthWidth = 78;
    const deleteWidth = 26;
    const selectWidth = Math.max(40, width - margin * 2 - toggleWidth - strengthWidth - deleteWidth - gap * 3);
    const toggleX = margin;
    const selectX = toggleX + toggleWidth + gap;
    const strengthX = selectX + selectWidth + gap;
    const deleteX = strengthX + strengthWidth + gap;
    return { toggleWidth, selectWidth, strengthWidth, deleteWidth, toggleX, selectX, strengthX, deleteX };
}

function fitSize(node) {
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0] || 0, DEFAULT_WIDTH), size[1]]);
}

function normalizeValue(value) {
    const source = value && typeof value === "object" ? value : {};
    const strength = Number(source.strength);
    return {
        on: source.on !== false,
        lora: typeof source.lora === "string" ? source.lora : "",
        strength: Number.isFinite(strength) ? strength : 1.0,
    };
}

function ensureSearchStyles(doc) {
    if (doc.getElementById(SEARCH_STYLE_ID)) {
        return;
    }
    const style = doc.createElement("style");
    style.id = SEARCH_STYLE_ID;
    style.textContent = `
        .anima-lora-search-overlay {
            position: fixed;
            inset: 0;
            z-index: 100000;
            background: transparent;
            color-scheme: dark;
        }
        .anima-lora-search-panel {
            position: fixed;
            display: flex;
            flex-direction: column;
            width: min(520px, calc(100vw - 24px));
            max-height: min(620px, calc(100vh - 24px));
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 9px;
            background: #202126;
            color: #f0f1f4;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.38);
            font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            transform: none !important;
        }
        .anima-lora-search-head {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 8px;
            padding: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.10);
        }
        .anima-lora-search-input {
            min-width: 0;
            height: 34px;
            box-sizing: border-box;
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 6px;
            outline: none;
            padding: 0 10px;
            background: #15161a;
            color: #f5f6f8;
            font: inherit;
        }
        .anima-lora-search-input:focus {
            border-color: #7bd88f;
            box-shadow: 0 0 0 1px #7bd88f;
        }
        .anima-lora-search-close {
            width: 34px;
            height: 34px;
            border: 0;
            border-radius: 6px;
            background: transparent;
            color: #aeb2bb;
            font: 18px/1 system-ui, sans-serif;
            cursor: pointer;
        }
        .anima-lora-search-close:hover,
        .anima-lora-search-close:focus-visible {
            background: rgba(255, 255, 255, 0.08);
            color: #fff;
            outline: none;
        }
        .anima-lora-search-status {
            padding: 6px 11px;
            color: #9da2ad;
            font-size: 11px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        }
        .anima-lora-search-results {
            flex: 1 1 auto;
            min-height: 42px;
            overflow: auto;
            overscroll-behavior: contain;
            padding: 5px;
        }
        .anima-lora-search-item {
            display: grid;
            width: 100%;
            box-sizing: border-box;
            gap: 1px;
            border: 0;
            border-radius: 5px;
            padding: 7px 9px;
            background: transparent;
            color: inherit;
            text-align: left;
            cursor: pointer;
        }
        .anima-lora-search-item:hover,
        .anima-lora-search-item[data-active="true"] {
            background: rgba(123, 216, 143, 0.14);
        }
        .anima-lora-search-name {
            overflow: hidden;
            color: #f1f2f5;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .anima-lora-search-path {
            overflow: hidden;
            color: #9197a2;
            font-size: 11px;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .anima-lora-search-empty {
            padding: 18px 10px;
            color: #9da2ad;
            text-align: center;
        }
    `;
    doc.head.append(style);
}

function searchableText(value) {
    return value.replaceAll("\\", "/").toLocaleLowerCase();
}

function filterLoras(values, query) {
    const terms = searchableText(query).trim().split(/\s+/).filter(Boolean);
    if (!terms.length) {
        return values;
    }
    return values.filter((value) => {
        const candidate = searchableText(value);
        return terms.every((term) => candidate.includes(term));
    });
}

function splitLoraPath(value) {
    const normalized = value.replaceAll("\\", "/");
    const slash = normalized.lastIndexOf("/");
    return slash < 0
        ? { name: normalized, path: "" }
        : { name: normalized.slice(slash + 1), path: normalized.slice(0, slash) };
}

function chooseLora(node, widget, event) {
    const values = Array.isArray(node.__animaLoraCatalog) ? node.__animaLoraCatalog : [];
    if (!values.length) {
        app.ui.dialog.show("没有发现 LoRA。请把文件放入 ComfyUI/models/loras 后刷新节点。", "Anima LoRA Stack");
        return;
    }

    activeLoraChooser?.close();
    const doc = event?.target?.ownerDocument || document;
    ensureSearchStyles(doc);

    const overlay = doc.createElement("div");
    overlay.className = "anima-lora-search-overlay";
    const panel = doc.createElement("section");
    panel.className = "anima-lora-search-panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.setAttribute("aria-label", "搜索并选择 Anima LoRA");

    const head = doc.createElement("div");
    head.className = "anima-lora-search-head";
    const input = doc.createElement("input");
    input.className = "anima-lora-search-input";
    input.type = "search";
    input.placeholder = "搜索文件名或子目录…";
    input.autocomplete = "off";
    input.spellcheck = false;
    const closeButton = doc.createElement("button");
    closeButton.className = "anima-lora-search-close";
    closeButton.type = "button";
    closeButton.title = "关闭";
    closeButton.setAttribute("aria-label", "关闭 LoRA 搜索");
    closeButton.textContent = "\u00d7";
    head.append(input, closeButton);

    const status = doc.createElement("div");
    status.className = "anima-lora-search-status";
    const results = doc.createElement("div");
    results.className = "anima-lora-search-results";
    results.setAttribute("role", "listbox");
    panel.append(head, status, results);
    overlay.append(panel);

    let visibleValues = [];
    let activeIndex = 0;
    let closed = false;

    const close = () => {
        if (closed) {
            return;
        }
        closed = true;
        overlay.remove();
        if (activeLoraChooser?.overlay === overlay) {
            activeLoraChooser = null;
        }
        app.canvas?.canvas?.focus?.();
    };

    const choose = (value) => {
        widget.value = { ...widget.value, lora: value };
        markChanged(node);
        close();
    };

    const setActive = (index, scroll = true) => {
        if (!visibleValues.length) {
            activeIndex = -1;
            return;
        }
        activeIndex = Math.max(0, Math.min(index, visibleValues.length - 1));
        const items = results.querySelectorAll(".anima-lora-search-item");
        items.forEach((item, itemIndex) => {
            const selected = itemIndex === activeIndex;
            item.dataset.active = String(selected);
            item.setAttribute("aria-selected", String(selected));
        });
        if (scroll) {
            items[activeIndex]?.scrollIntoView({ block: "nearest" });
        }
    };

    const render = () => {
        const matched = filterLoras(values, input.value);
        visibleValues = matched.slice(0, SEARCH_RESULT_LIMIT);
        results.replaceChildren();
        const suffix = matched.length > SEARCH_RESULT_LIMIT
            ? `，仅显示前 ${SEARCH_RESULT_LIMIT} 条，请继续输入缩小范围`
            : "";
        status.textContent = `找到 ${matched.length} / ${values.length} 个 LoRA${suffix}`;

        if (!visibleValues.length) {
            const empty = doc.createElement("div");
            empty.className = "anima-lora-search-empty";
            empty.textContent = "没有匹配的 LoRA";
            results.append(empty);
            activeIndex = -1;
            return;
        }

        const fragment = doc.createDocumentFragment();
        visibleValues.forEach((value, index) => {
            const parts = splitLoraPath(value);
            const item = doc.createElement("button");
            item.className = "anima-lora-search-item";
            item.type = "button";
            item.setAttribute("role", "option");
            item.title = value;
            const name = doc.createElement("span");
            name.className = "anima-lora-search-name";
            name.textContent = parts.name;
            item.append(name);
            if (parts.path) {
                const itemPath = doc.createElement("span");
                itemPath.className = "anima-lora-search-path";
                itemPath.textContent = parts.path;
                item.append(itemPath);
            }
            item.addEventListener("pointerenter", () => setActive(index, false));
            item.addEventListener("click", () => choose(value));
            fragment.append(item);
        });
        results.append(fragment);
        const selectedIndex = visibleValues.indexOf(widget.value.lora);
        setActive(selectedIndex >= 0 ? selectedIndex : 0, false);
    };

    const stopCanvasEvent = (currentEvent) => currentEvent.stopPropagation();
    for (const type of ["pointerdown", "mousedown", "mouseup", "click", "contextmenu", "wheel"]) {
        panel.addEventListener(type, stopCanvasEvent);
    }
    overlay.addEventListener("pointerdown", (currentEvent) => {
        if (currentEvent.target === overlay) {
            close();
        }
    });
    closeButton.addEventListener("click", close);
    input.addEventListener("input", render);
    input.addEventListener("keydown", (currentEvent) => {
        currentEvent.stopPropagation();
        if (currentEvent.key === "Escape") {
            currentEvent.preventDefault();
            close();
        } else if (currentEvent.key === "ArrowDown") {
            currentEvent.preventDefault();
            setActive(activeIndex + 1);
        } else if (currentEvent.key === "ArrowUp") {
            currentEvent.preventDefault();
            setActive(activeIndex - 1);
        } else if (currentEvent.key === "Enter" && activeIndex >= 0) {
            currentEvent.preventDefault();
            choose(visibleValues[activeIndex]);
        }
    });

    const root = doc.fullscreenElement || doc.body;
    root.append(overlay);
    activeLoraChooser = { overlay, close };
    render();

    const viewport = doc.defaultView || window;
    const panelRect = panel.getBoundingClientRect();
    const requestedX = Number.isFinite(event?.clientX)
        ? event.clientX + 8
        : (viewport.innerWidth - panelRect.width) / 2;
    const requestedY = Number.isFinite(event?.clientY)
        ? event.clientY + 8
        : (viewport.innerHeight - panelRect.height) / 2;
    panel.style.left = `${Math.max(12, Math.min(requestedX, viewport.innerWidth - panelRect.width - 12))}px`;
    panel.style.top = `${Math.max(12, Math.min(requestedY, viewport.innerHeight - panelRect.height - 12))}px`;
    input.focus({ preventScroll: true });
}

function editStrength(node, widget) {
    const initial = Number(widget.value.strength).toFixed(2);
    const raw = window.prompt("MODEL 强度（支持负数）", initial);
    if (raw === null) {
        return;
    }
    const strength = Number(raw);
    if (!Number.isFinite(strength)) {
        app.ui.dialog.show("强度必须是有限数字。", "Anima LoRA Stack");
        return;
    }
    widget.value = { ...widget.value, strength };
    markChanged(node);
}

function moveRow(node, widget, offset) {
    const widgets = node.widgets || [];
    const current = widgets.indexOf(widget);
    if (current < 0) {
        return;
    }
    let target = current + offset;
    while (target >= 0 && target < widgets.length && !widgets[target].name?.startsWith(ROW_PREFIX)) {
        target += offset;
    }
    if (target < 0 || target >= widgets.length) {
        return;
    }
    [widgets[current], widgets[target]] = [widgets[target], widgets[current]];
    renumberRows(node);
    fitSize(node);
    markChanged(node);
}

function removeRow(node, widget) {
    const index = node.widgets?.indexOf(widget) ?? -1;
    if (index >= 0) {
        node.widgets.splice(index, 1);
        renumberRows(node);
        fitSize(node);
        markChanged(node);
    }
}

function showRowMenu(node, widget, event) {
    const rows = rowWidgets(node);
    const index = rows.indexOf(widget);
    new LiteGraph.ContextMenu([
        {
            content: widget.value.on ? "禁用" : "启用",
            callback: () => {
                widget.value = { ...widget.value, on: !widget.value.on };
                markChanged(node);
            },
        },
        null,
        {
            content: "上移",
            disabled: index <= 0,
            callback: () => moveRow(node, widget, -1),
        },
        {
            content: "下移",
            disabled: index < 0 || index >= rows.length - 1,
            callback: () => moveRow(node, widget, 1),
        },
        null,
        {
            content: "删除",
            callback: () => removeRow(node, widget),
        },
    ], { event, title: "LoRA 条目" });
}

function makeRowWidget(node, value = undefined, name = undefined) {
    let currentValue = normalizeValue(value);
    const widget = {
        type: "custom",
        name: name || nextRowName(node),
        options: { serialize: true },
        get value() {
            return currentValue;
        },
        set value(next) {
            currentValue = normalizeValue(next);
        },
        computeSize(width) {
            return [width || DEFAULT_WIDTH, 28];
        },
        serializeValue() {
            return { ...currentValue };
        },
        draw(ctx, owner, width, y, height) {
            const rowHeight = Math.min(height || 28, 26);
            const rowWidth = owner.size?.[0] || width || DEFAULT_WIDTH;
            const { toggleWidth, selectWidth, strengthWidth, deleteWidth, toggleX, selectX, strengthX, deleteX } = columnLayout(rowWidth);
            const midY = y + rowHeight / 2;

            widget.__hitAreas = {
                toggle: [toggleX, y, toggleWidth, rowHeight],
                select: [selectX, y, selectWidth, rowHeight],
                strength: [strengthX, y, strengthWidth, rowHeight],
                delete: [deleteX, y, deleteWidth, rowHeight],
            };

            ctx.save();
            ctx.globalAlpha = currentValue.on ? 1 : 0.48;
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
            ctx.lineWidth = canvasLineWidth();

            for (const bounds of Object.values(widget.__hitAreas)) {
                ctx.beginPath();
                ctx.roundRect(bounds[0], bounds[1], bounds[2], bounds[3], 7);
                ctx.fill();
                ctx.stroke();
            }

            ctx.fillStyle = currentValue.on ? "#7bd88f" : "#777";
            ctx.beginPath();
            ctx.arc(toggleX + toggleWidth / 2, midY, 6, 0, Math.PI * 2);
            ctx.fill();

            if (!isLowQuality()) {
                ctx.font = "12px sans-serif";
                ctx.textBaseline = "middle";
                ctx.textAlign = "left";
                ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
                const label = currentValue.lora || "选择 LoRA…";
                ctx.fillText(fitText(ctx, label, selectWidth - 20), selectX + 10, midY);

                ctx.textAlign = "center";
                ctx.fillText(Number(currentValue.strength).toFixed(2), strengthX + strengthWidth / 2, midY);

                const deleteCenterX = deleteX + deleteWidth / 2;
                ctx.strokeStyle = "#c25757";
                ctx.lineWidth = canvasLineWidth() * 2;
                ctx.lineCap = "round";
                ctx.beginPath();
                ctx.moveTo(deleteCenterX - 4, midY - 4);
                ctx.lineTo(deleteCenterX + 4, midY + 4);
                ctx.moveTo(deleteCenterX + 4, midY - 4);
                ctx.lineTo(deleteCenterX - 4, midY + 4);
                ctx.stroke();
            }
            ctx.restore();
        },
        mouse(event, pos, owner) {
            if (event.type !== "pointerdown" && event.type !== "mousedown") {
                return false;
            }
            const [x, y] = pos;
            if (event.button === 2) {
                showRowMenu(owner, widget, event);
                return true;
            }
            const hit = (bounds) => bounds && x >= bounds[0] && x <= bounds[0] + bounds[2]
                && y >= bounds[1] && y <= bounds[1] + bounds[3];
            if (hit(widget.__hitAreas?.toggle)) {
                widget.value = { ...currentValue, on: !currentValue.on };
                markChanged(owner);
                return true;
            }
            if (hit(widget.__hitAreas?.select)) {
                chooseLora(owner, widget, event);
                return true;
            }
            if (hit(widget.__hitAreas?.strength)) {
                editStrength(owner, widget);
                return true;
            }
            if (hit(widget.__hitAreas?.delete)) {
                removeRow(owner, widget);
                return true;
            }
            return false;
        },
    };
    node.addCustomWidget(widget);
    const addButtonIndex = node.widgets?.findIndex((item) => item.name === "+ Add LoRA") ?? -1;
    const widgetIndex = node.widgets?.indexOf(widget) ?? -1;
    if (addButtonIndex >= 0 && widgetIndex > addButtonIndex) {
        node.widgets.splice(widgetIndex, 1);
        node.widgets.splice(addButtonIndex, 0, widget);
    }
    return widget;
}

function addHeaderWidget(node) {
    const widget = {
        type: "custom",
        name: "anima_lora_header",
        options: { serialize: false },
        computeSize(width) {
            return [width || DEFAULT_WIDTH, 24];
        },
        draw(ctx, owner, width, y, height) {
            const rows = rowWidgets(owner);
            const allOn = rows.length > 0 && rows.every((row) => row.value.on);
            const mixed = rows.some((row) => row.value.on) && !allOn;
            const rowHeight = Math.min(height || 28, 26);
            const rowWidth = owner.size?.[0] || width || DEFAULT_WIDTH;
            const { toggleWidth, selectWidth, strengthWidth, toggleX, selectX, strengthX } = columnLayout(rowWidth);
            const midY = y + rowHeight / 2;
            widget.__toggleBounds = [toggleX, y, toggleWidth, rowHeight];
            ctx.save();
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.beginPath();
            ctx.roundRect(...widget.__toggleBounds, 7);
            ctx.fill();
            ctx.lineWidth = canvasLineWidth();
            ctx.stroke();
            ctx.fillStyle = mixed ? "#d6b45c" : (allOn ? "#7bd88f" : "#777");
            ctx.beginPath();
            ctx.arc(toggleX + toggleWidth / 2, midY, 6, 0, Math.PI * 2);
            ctx.fill();
            if (!isLowQuality()) {
                ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || LiteGraph.WIDGET_TEXT_COLOR;
                ctx.font = "12px sans-serif";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillText("全部启用/禁用", selectX + 10, midY);
                ctx.textAlign = "center";
                ctx.fillText("MODEL Strength", strengthX + strengthWidth / 2, midY);
            }
            ctx.restore();
        },
        mouse(event, pos, owner) {
            if (event.type !== "pointerdown" && event.type !== "mousedown") {
                return false;
            }
            const bounds = widget.__toggleBounds;
            if (!bounds) {
                return false;
            }
            const [x, y] = pos;
            if (x < bounds[0] || x > bounds[0] + bounds[2] || y < bounds[1] || y > bounds[1] + bounds[3]) {
                return false;
            }
            const rows = rowWidgets(owner);
            const turnOn = !rows.length || !rows.every((row) => row.value.on);
            rows.forEach((row) => {
                row.value = { ...row.value, on: turnOn };
            });
            markChanged(owner);
            return true;
        },
    };
    node.addCustomWidget(widget);
}

function addButton(node) {
    node.addWidget("button", "+ Add LoRA", null, () => {
        makeRowWidget(node);
        renumberRows(node);
        fitSize(node);
        markChanged(node);
    }, { serialize: false });
}

app.registerExtension({
    name: "anima.28-to-40-power-lora-stack",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) {
            return;
        }

        const originalComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function (...args) {
            const size = [...originalComputeSize.apply(this, args)];
            size[0] = Math.max(size[0], DEFAULT_WIDTH);
            return size;
        };

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            this.title = "Anima 28→40 Power LoRA Stack";

            const catalogWidget = this.widgets?.find((widget) => widget.name === "_lora_catalog");
            this.__animaLoraCatalog = catalogWidget?.options?.values
                ? [...catalogWidget.options.values].filter(Boolean)
                : [];
            if (catalogWidget) {
                catalogWidget.type = "hidden";
                catalogWidget.hidden = true;
                catalogWidget.computeSize = () => [0, -4];
                catalogWidget.serializeValue = () => null;
            }

            addHeaderWidget(this);
            makeRowWidget(this);
            addButton(this);
            this.size[0] = DEFAULT_WIDTH;
            fitSize(this);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = originalConfigure?.apply(this, arguments);
            const serialized = Array.isArray(info?.widgets_values) ? info.widgets_values : [];
            const savedRows = serialized.filter((value) => value && typeof value === "object"
                && Object.hasOwn(value, "on") && Object.hasOwn(value, "lora")
                && Object.hasOwn(value, "strength"));

            if (savedRows.length) {
                const rows = rowWidgets(this);
                savedRows.forEach((value, index) => {
                    if (rows[index]) {
                        rows[index].value = value;
                    } else {
                        makeRowWidget(this, value);
                    }
                });
                while (rowWidgets(this).length > savedRows.length) {
                    removeRow(this, rowWidgets(this).at(-1));
                }
                renumberRows(this);
            }
            fitSize(this);
            return result;
        };
    },
});

