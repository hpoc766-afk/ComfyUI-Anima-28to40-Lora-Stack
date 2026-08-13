import { app } from "../../scripts/app.js";

const NODE_CLASS = "Anima28To40PowerLoraStack";
const ROW_PREFIX = "lora_";
const DEFAULT_WIDTH = 430;

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

function normalizeValue(value) {
    const source = value && typeof value === "object" ? value : {};
    const strength = Number(source.strength);
    return {
        on: source.on !== false,
        lora: typeof source.lora === "string" ? source.lora : "",
        strength: Number.isFinite(strength) ? strength : 1.0,
    };
}

function chooseLora(node, widget, event) {
    const values = Array.isArray(node.__animaLoraCatalog) ? node.__animaLoraCatalog : [];
    if (!values.length) {
        app.ui.dialog.show("没有发现 LoRA。请把文件放入 ComfyUI/models/loras 后刷新节点。", "Anima LoRA Stack");
        return;
    }
    const menuItems = values.map((name) => ({
        content: name,
        callback: () => {
            widget.value = { ...widget.value, lora: name };
            markChanged(node);
        },
    }));
    new LiteGraph.ContextMenu(menuItems, {
        event,
        title: "选择 28 层 Anima LoRA",
        scale: Math.max(1, app.canvas?.ds?.scale || 1),
    });
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
    node.setSize(node.computeSize());
    markChanged(node);
}

function removeRow(node, widget) {
    const index = node.widgets?.indexOf(widget) ?? -1;
    if (index >= 0) {
        node.widgets.splice(index, 1);
        renumberRows(node);
        node.setSize(node.computeSize());
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
            const margin = 10;
            const gap = 6;
            const rowHeight = Math.min(height || 28, 26);
            const toggleWidth = 34;
            const strengthWidth = 78;
            const selectWidth = Math.max(120, width - margin * 2 - toggleWidth - strengthWidth - gap * 2);
            const toggleX = margin;
            const selectX = toggleX + toggleWidth + gap;
            const strengthX = selectX + selectWidth + gap;
            const midY = y + rowHeight / 2;

            widget.__hitAreas = {
                toggle: [toggleX, y, toggleWidth, rowHeight],
                select: [selectX, y, selectWidth, rowHeight],
                strength: [strengthX, y, strengthWidth, rowHeight],
            };

            ctx.save();
            ctx.globalAlpha = currentValue.on ? 1 : 0.48;
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
            ctx.lineWidth = 1;

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

            ctx.font = "12px sans-serif";
            ctx.textBaseline = "middle";
            ctx.textAlign = "left";
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
            const label = currentValue.lora || "选择 LoRA…";
            ctx.fillText(fitText(ctx, label, selectWidth - 20), selectX + 10, midY);

            ctx.textAlign = "center";
            ctx.fillText(Number(currentValue.strength).toFixed(2), strengthX + strengthWidth / 2, midY);
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
            widget.__toggleBounds = [10, y + 2, 34, Math.min(20, height)];
            ctx.save();
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.beginPath();
            ctx.roundRect(...widget.__toggleBounds, 7);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = mixed ? "#d6b45c" : (allOn ? "#7bd88f" : "#777");
            ctx.beginPath();
            ctx.arc(27, y + 12, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || LiteGraph.WIDGET_TEXT_COLOR;
            ctx.font = "12px sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText("全部启用/禁用", 52, y + 12);
            ctx.textAlign = "center";
            ctx.fillText("MODEL Strength", width - 56, y + 12);
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
        node.setSize(node.computeSize());
        markChanged(node);
    }, { serialize: false });
}

app.registerExtension({
    name: "anima.28-to-40-power-lora-stack",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) {
            return;
        }

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            this.title = "Anima 28→40 Power LoRA Stack";
            this.size[0] = Math.max(this.size[0], DEFAULT_WIDTH);

            const catalogWidget = this.widgets?.find((widget) => widget.name === "_lora_catalog");
            this.__animaLoraCatalog = catalogWidget?.options?.values
                ? [...catalogWidget.options.values].filter(Boolean)
                : [];
            if (catalogWidget) {
                catalogWidget.type = "hidden";
                catalogWidget.computeSize = () => [0, -4];
                catalogWidget.serializeValue = () => null;
            }

            addHeaderWidget(this);
            makeRowWidget(this);
            addButton(this);
            this.setSize(this.computeSize());
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
                this.setSize(this.computeSize());
            }
            return result;
        };
    },
});

