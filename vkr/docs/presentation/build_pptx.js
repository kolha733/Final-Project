// Презентация к защите ВКР (.pptx).
//
// Ответственные: Будаев К. В. (слайды 1–8), Гарифзянов Т. Р. (слайды 9–15).
//
// Числа и ряды диаграмм берутся из результатов экспериментов через
// slide_data.py (тот же модуль facts.py, что и в пояснительной записке),
// рисунки — из docs/figures/zapiska/ (копии без общего заголовка).
// После прогона FULL презентация пересобирается без ручной правки чисел.
//
// Запуск (из каталога vkr):
//   (cd docs/presentation && npm ci)      # один раз: pptxgenjs из package-lock.json
//   node docs/presentation/build_pptx.js

"use strict";

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const ROOT = path.resolve(HERE, "..", "..");
const OUT = path.join(HERE, "VKR_Budaev_Garifzyanov_presentation.pptx");
const FIG = (name) => path.join(ROOT, "docs", "figures", "zapiska", name);

// --- данные ---------------------------------------------------------------

const python = process.env.PYTHON || "python";
const DATA = JSON.parse(
	execFileSync(python, [path.join(HERE, "slide_data.py")], { cwd: ROOT, encoding: "utf8", maxBuffer: 1 << 26 }),
);
const F = DATA.facts;
const S = DATA.series;
for (const [key, value] of Object.entries(F)) {
	if (value === undefined || value === null) throw new Error(`нет значения ${key}`);
}
const fact = (key) => {
	if (!(key in F)) throw new Error(`нет значения ${key} в facts.py`);
	return String(F[key]);
};

// --- тема -------------------------------------------------------------------
// Цвета согласованы с рисунками работы (scendrift.reporting.style) и со схемами
// архитектуры: синий — Будаев К. В., зелёный — Гарифзянов Т. Р.

const THEME = {
	name: "scendrift",
	headFontFace: "Cambria",
	bodyFontFace: "Calibri",
	colors: {
		dk1: "1F2328", // основной текст
		lt1: "FFFFFF",
		dk2: "14213D", // тёмный фон титульного и итогового слайдов
		lt2: "EEF2F7", // фон карточек
		accent1: "2A78D6", // синий рисунков: детекторы, «реальный дрейф»
		accent2: "EB6834", // оранжевый рисунков: выделение (Periodic, «виртуальный дрейф»)
		accent3: "1BAF7A", // аква рисунков: выбранный вариант
		accent4: "5B6573", // приглушённый текст подписей
		accent5: "2B5C8A", // Будаев К. В. (как на схемах архитектуры)
		accent6: "2E7D4F", // Гарифзянов Т. Р.
		hlink: "2A78D6",
		folHlink: "4A3AA7",
	},
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13,333 × 7,5 дюйма
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Будаев К. В., Гарифзянов Т. Р.";
pres.company = "Финансовый университет при Правительстве Российской Федерации";
pres.title = "Разработка фреймворка генерации бенчмарков систем выявления концептуального дрейфа";
const C = pres.SchemeColor;

// --- геометрия --------------------------------------------------------------

const W = 13.333;
const M = 0.6; // поле
const CW = W - 2 * M; // ширина области содержимого
const TOP = 1.65; // начало области содержимого под заголовком
const GAP = 0.3;

// --- макеты -----------------------------------------------------------------

const SPEAKERS = {
	K: { name: "Будаев К. В.", color: C.accent5 },
	T: { name: "Гарифзянов Т. Р.", color: C.accent6 },
};

function titlePlaceholder(color) {
	return {
		placeholder: {
			options: {
				name: "title", type: "title", x: M, y: 0.38, w: CW, h: 1.05,
				fontFace: THEME.headFontFace, fontSize: 28, bold: true, color, align: "left", valign: "top", margin: 0,
			},
			text: "",
		},
	};
}

function footer(speaker, dark) {
	const s = SPEAKERS[speaker];
	return [
		{ ellipse: { x: M, y: 7.0, w: 0.14, h: 0.14, fill: { color: dark ? C.background1 : s.color }, line: { type: "none" } } },
		{ text: { text: `Докладчик: ${s.name}`, options: { x: M + 0.22, y: 6.9, w: 5, h: 0.34, fontSize: 11, margin: 0, color: dark ? C.background2 : C.accent4, valign: "middle" } } },
	];
}

for (const speaker of ["K", "T"]) {
	pres.defineSlideMaster({
		title: `CONTENT_${speaker}`,
		background: { color: C.background1 },
		objects: [titlePlaceholder(C.text2), ...footer(speaker, false)],
		slideNumber: { x: W - M - 0.6, y: 6.9, w: 0.6, h: 0.34, fontSize: 11, color: C.accent4, align: "right" },
	});
}
pres.defineSlideMaster({
	title: "DARK_T",
	background: { color: C.text2 },
	objects: [titlePlaceholder(C.background1), ...footer("T", true)],
	slideNumber: { x: W - M - 0.6, y: 6.9, w: 0.6, h: 0.34, fontSize: 11, color: C.background2, align: "right" },
});
pres.defineSlideMaster({
	title: "TITLE",
	background: { color: C.text2 },
	objects: [
		{ text: { text: "Финансовый университет при Правительстве Российской Федерации\nФакультет информационных технологий и анализа больших данных · Кафедра искусственного интеллекта",
			options: { x: M, y: 0.45, w: CW, h: 0.8, fontSize: 13, color: C.background2, margin: 0, valign: "top" } } },
		{ placeholder: { options: { name: "title", type: "title", x: M, y: 1.45, w: 11.4, h: 1.75, fontFace: THEME.headFontFace,
			fontSize: 28, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
		{ placeholder: { options: { name: "kind", type: "body", x: M, y: 3.45, w: CW, h: 0.45, fontSize: 16,
			color: C.background2, align: "left", margin: 0, valign: "top" }, text: "" } },
		{ placeholder: { options: { name: "people", type: "body", x: M, y: 4.25, w: 7.2, h: 1.7, fontSize: 15,
			color: C.background1, align: "left", margin: 0, valign: "top" }, text: "" } },
		{ text: { text: "Москва — 2027", options: { x: M, y: 6.75, w: 4, h: 0.4, fontSize: 13, color: C.background2, margin: 0 } } },
	],
});

// --- элементы слайдов -------------------------------------------------------

let objectCounter = 0;
const oname = (kind) => `${kind}-${++objectCounter}`;

function card(slide, x, y, w, h, fill = C.background2) {
	slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
		x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { type: "none" }, objectName: oname("card"),
	});
}

function circleNumber(slide, n, x, y, color, d = 0.5) {
	slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { type: "none" }, objectName: oname("marker") });
	slide.addText(String(n), {
		x, y, w: d, h: d, align: "center", valign: "middle", fontSize: 16, bold: true, color: C.background1, margin: 0,
		isTextBox: true, objectName: oname("marker-text"),
	});
}

// Крупное число и подпись под ним (внутри карточки или без неё).
function stat(slide, x, y, w, value, label, { color = C.accent1, valueSize = 34, labelSize = 14, h = 1.45, onDark = false } = {}) {
	slide.addText(value, {
		x, y, w, h: 0.62, fontSize: valueSize, bold: true, color, margin: 0, valign: "bottom", isTextBox: true,
		fontFace: THEME.headFontFace, objectName: oname("stat-value"),
	});
	slide.addText(label, {
		x, y: y + 0.68, w, h: h - 0.68, fontSize: labelSize, color: onDark ? C.background2 : C.text1, margin: 0, valign: "top",
		isTextBox: true, objectName: oname("stat-label"),
	});
}

function text(slide, value, x, y, w, h, opts = {}) {
	slide.addText(value, { x, y, w, h, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true,
		objectName: oname("text"), ...opts });
}

function bullets(slide, items, x, y, w, h, opts = {}) {
	slide.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })), {
		x, y, w, h, fontSize: 16, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 10, isTextBox: true,
		objectName: oname("bullets"), ...opts,
	});
}

function image(slide, file, x, y, w, h, alt) {
	if (!fs.existsSync(file)) throw new Error(`нет рисунка ${file}`);
	slide.addImage({ path: file, x, y, w, h, altText: alt, objectName: oname("figure") });
}

// Высота рисунка при заданной ширине (по размерам PNG).
function pngHeight(file, w) {
	const buf = fs.readFileSync(file);
	const pw = buf.readUInt32BE(16);
	const ph = buf.readUInt32BE(20);
	return (w * ph) / pw;
}

// Общий вид диаграмм: одинаковые шрифты, подписи и сетка на всех слайдах.
function chartStyle(extra = {}) {
	return {
		catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
		titleFontFace: "+mn-lt", legendFontFace: "+mn-lt",
		catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 11, legendFontSize: 12, titleFontSize: 14,
		catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent4, dataLabelColor: HEX.dk1, titleColor: HEX.dk1,
		valGridLine: { color: "D9DEE5", size: 0.5 }, catGridLine: { style: "none" },
		catAxisLineShow: false, valAxisLineShow: false,
		showValue: true, dataLabelPosition: "outEnd",
		...extra,
	};
}

// Заметки докладчика с оценкой времени (~120 слов в минуту).
const timing = [];
function say(slide, who, body) {
	const words = (body.match(/[\p{L}\p{N}]+/gu) || []).length;
	const seconds = Math.max(15, Math.round(words / 2 / 5) * 5);
	slide.addNotes(`[${who}, ~${seconds} с] ${body}`);
	timing.push({ who, seconds });
}

function addSlide(layout, section, title) {
	const slide = pres.addSlide({ masterName: layout, sectionTitle: section });
	if (title) slide.addText(title, { placeholder: "title" });
	return slide;
}

const SECTIONS = ["Постановка задачи", "Модель и метод", "Фреймворк и эксперименты", "Итоги"];

// =============================================================================
// 1. Титульный слайд (Будаев К. В.)
// =============================================================================

pres.addSection({ title: SECTIONS[0] });
{
	const s = pres.addSlide({ masterName: "TITLE", sectionTitle: SECTIONS[0] });
	s.addText("Разработка фреймворка генерации бенчмарков систем выявления концептуального дрейфа с автоматизированным формированием и параметризацией сценариев", { placeholder: "title" });
	s.addText("Выпускная квалификационная работа (коллективная) · направление 01.03.02 «Прикладная математика и информатика»", { placeholder: "kind" });
	s.addText([
		{ text: "Выполнили студенты группы ПМ23-5:", options: { color: C.background2, breakLine: true } },
		{ text: "Будаев Константин Владимирович", options: { bold: true, breakLine: true } },
		{ text: "Гарифзянов Тимур Русланович", options: { bold: true, breakLine: true } },
		{ text: "Научный руководитель: ассистент кафедры ИИ Окунева Эвелина Александровна", options: { color: C.background2 } },
	], { placeholder: "people" });
	// Мотив «поток с дрейфом»: объекты потока, концепт которых сменяется.
	const n = 22;
	for (let i = 0; i < n; i++) {
		const p = Math.min(1, Math.max(0, (i - 8) / 6));
		const y = 5.0 - 0.55 * p;
		s.addShape(pres.shapes.OVAL, {
			x: 8.3 + i * 0.2, y, w: 0.12, h: 0.12, line: { type: "none" },
			fill: { color: p < 0.5 ? C.accent1 : C.accent2 }, objectName: oname("stream-dot"),
		});
	}
	s.addText("концепт A → концепт B", { x: 8.3, y: 5.35, w: 4.4, h: 0.35, fontSize: 12, color: C.background2, margin: 0,
		align: "right", isTextBox: true, objectName: oname("text") });
	say(s, "Будаев К. В.",
		"Здравствуйте, уважаемые члены комиссии. Мы представляем коллективную выпускную работу " +
		"о генерации бенчмарков для детекторов концептуального дрейфа. Я, Константин Будаев, расскажу о постановке задачи, " +
		"формальной модели сценария и методе формирования наборов. Тимур Гарифзянов расскажет о реализации фреймворка " +
		"и результатах экспериментов.",
	);
}

// =============================================================================
// 2. Проблема (Будаев К. В.)
// =============================================================================
{
	const s = addSlide("CONTENT_K", SECTIONS[0], "Детекторы дрейфа сравнивают на потоках с неконтролируемой величиной дрейфа");
	bullets(s, [
		"Модели в эксплуатации теряют качество из-за концептуального дрейфа — изменения распределения P(X, y) во времени",
		"Детекторов предложено много: DDM, EDDM, ADWIN, KSWIN, Page–Hinkley, HDDM, FHDDM",
		"Сравнивают их на нескольких вручную составленных потоках: SEA, Agrawal, STAGGER",
		"Нельзя ответить, при какой величине и скорости дрейфа детектор надёжно работает",
	], M, TOP + 0.1, 6.6, 4.8, { fontSize: 18, paraSpaceAfter: 14 });
	const x = M + 6.6 + 0.5;
	const w = CW - 6.6 - 0.5;
	card(s, x, TOP + 0.1, w, 4.6, C.text2);
	text(s, "Величина смены концепта у классических генераторов river (расстояние полной вариации)", x + 0.4, TOP + 0.45, w - 0.8, 0.9,
		{ fontSize: 15, color: C.background2 });
	text(s, `${fact("e1.classic_min")} – ${fact("e1.classic_max")}`, x + 0.4, TOP + 1.35, w - 0.8, 1.1,
		{ fontSize: 54, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "middle" });
	text(s, `лишь ${fact("e1.classic_small")} из ${fact("e1.classic_n")} смен концепта меньше 0,25: малые и средние изменения ручные бенчмарки почти не покрывают`,
		x + 0.4, TOP + 2.65, w - 0.8, 1.6, { fontSize: 15, color: C.background2 });
	say(s, "Будаев К. В.",
		`Модели машинного обучения в эксплуатации теряют качество, когда меняется распределение данных, — это концептуальный дрейф. ` +
		`Для его обнаружения предложено много детекторов, но сравнивают их на нескольких вручную составленных потоках. ` +
		`Мы измерили величину смены концепта у классических генераторов river на единой шкале: она неконтролируема и лежит от ${fact("e1.classic_min")} до ${fact("e1.classic_max")}, ` +
		`а малые изменения почти не представлены. Поэтому по таким бенчмаркам нельзя понять, при какой величине и скорости дрейфа детектор работает надёжно.`,
	);
}

// =============================================================================
// 3. Пробелы существующих средств (Будаев К. В.)
// =============================================================================
{
	const s = addSlide("CONTENT_K", SECTIONS[0], "Ни одно существующее средство не решает задачу целиком");
	const head = ["Средство", "Величина с измеримым смыслом", "Проверка допустимости сценария", "Автоматическое формирование наборов",
		"Метрики обнаружения", "Статистическое сравнение"];
	const rows = [
		["MOA и CD-MOA", "±", "±", "±", "±", "+"],
		["river", "±", "±", "–", "–", "–"],
		["CapyMOA", "±", "±", "–", "+", "–"],
		["stream-learn", "–", "±", "–", "±", "–"],
		["генератор Петижана–Уэбба", "+", "±", "–", "–", "–"],
		["scendrift (наша работа)", "+", "+", "+", "+", "+"],
	];
	const cell = (t, opts = {}) => ({ text: t, options: { align: "center", valign: "middle", ...opts } });
	const table = [head.map((h, i) => cell(h, { bold: true, color: C.background1, fill: { color: C.text2 }, align: i ? "center" : "left", fontSize: 13 }))];
	rows.forEach((r, ri) => {
		const ours = ri === rows.length - 1;
		table.push(r.map((t, i) => cell(t, {
			align: i ? "center" : "left", bold: ours, fontSize: i ? 20 : 15,
			color: ours ? C.background1 : (t === "+" ? C.accent3 : t === "–" ? C.accent2 : C.text1),
			fill: { color: ours ? C.accent1 : (ri % 2 ? C.background1 : C.background2) },
		})));
	});
	s.addTable(table, {
		x: M, y: TOP + 0.05, w: CW, colW: [3.13, 1.8, 1.8, 1.8, 1.8, 1.8], rowH: [0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
		border: { type: "solid", pt: 0.75, color: "FFFFFF" }, fontFace: THEME.bodyFontFace, margin: 0.06, objectName: oname("table"),
	});
	text(s, "+ есть   ± частично   – нет   ·   по исходному коду и документации актуальных версий, октябрь 2026 г.",
		M, TOP + 3.95, CW, 0.4, { fontSize: 12, color: C.accent4 });
	text(s, "Пять пробелов: нет общей шкалы величины, сценарий не проверяется, наборы составляют вручную, генерация и оценка разнесены по инструментам, области отказа изучают по одному параметру",
		M, TOP + 4.4, CW, 0.75, { fontSize: 15 });
	say(s, "Будаев К. В.",
		"Мы сравнили существующие средства по исходному коду и документации. " +
		"MOA с модулем CD-MOA даёт метрики и статистику, но величина дрейфа там — внутренний параметр генератора. " +
		"CapyMOA описывает сценарий и считает метрики, но наборы не формирует. Генератор Петижана–Уэбба калибрует величину, " +
		"но только для категориальных данных и внезапного дрейфа. Ни одно средство не проверяет допустимость сценария полностью и не формирует наборы по плану эксперимента — " +
		"это и есть пробелы, которые закрывает работа.",
	);
}

// =============================================================================
// 4. Цель и задачи (Будаев К. В.)
// =============================================================================
{
	const s = addSlide("CONTENT_K", SECTIONS[0], "Цель: фреймворк, формирующий сценарии дрейфа автоматически и с проверяемым смыслом");
	const items = [
		["Формальная модель сценария", "кортеж параметров, ограничения допустимости, единая шкала величины дрейфа"],
		["Метод формирования наборов", "планы эксперимента над пространством параметров с учётом ограничений"],
		["Фреймворк scendrift", "генерация потоков с истинной разметкой, прогон детекторов, метрики, статистика, отчёты"],
		["Экспериментальное исследование", "проверка генератора и сравнение девяти детекторов river в восьми экспериментах"],
	];
	const cw = (CW - 3 * GAP) / 4;
	items.forEach(([head, body], i) => {
		const x = M + i * (cw + GAP);
		card(s, x, TOP + 0.15, cw, 3.55);
		circleNumber(s, i + 1, x + 0.3, TOP + 0.45, C.accent5);
		text(s, head, x + 0.3, TOP + 1.15, cw - 0.6, 0.95, { fontSize: 18, bold: true, color: C.text2 });
		text(s, body, x + 0.3, TOP + 2.2, cw - 0.6, 1.4, { fontSize: 15 });
	});
	text(s, "Объект — методы выявления концептуального дрейфа в потоках данных. Предмет — методы и программные средства формирования параметризованных сценариев дрейфа для сравнительной оценки детекторов.",
		M, TOP + 4.0, CW, 0.9, { fontSize: 15, color: C.accent4 });
	say(s, "Будаев К. В.",
		"Цель работы — разработать фреймворк генерации бенчмарков, в котором сценарий дрейфа задаётся формализованным набором параметров, " +
		"а наборы сценариев формируются автоматически. Для этого решены четыре группы задач: построена формальная модель сценария, " +
		"разработан метод формирования наборов, реализован фреймворк scendrift и проведено экспериментальное исследование. " +
		"Объект исследования — методы выявления дрейфа, предмет — средства формирования сценариев для их оценки.",
	);
}

// =============================================================================
// 5. Модель сценария (Будаев К. В.)
// =============================================================================
pres.addSection({ title: SECTIONS[1] });
{
	const s = addSlide("CONTENT_K", SECTIONS[1], "Сценарий дрейфа — проверяемый объект S = ⟨B, E, Ω, s⟩");
	const parts = [
		["B", "поток", "семейство генератора, длина, размерность, доля класса π₀, шум меток η, структурный seed"],
		["E", "события дрейфа", "центр τ, ширина ℓ, форма φ, вид κ, величина m, доля признаков α, возврат ρ"],
		["Ω", "протокол оценки", "базовая модель, реакция на срабатывание, разогрев W, окно допуска Δ"],
		["s", "seed реализации", "выборка объектов и шум отделены от геометрии концептов"],
	];
	const cw = (CW - 3 * GAP) / 4;
	parts.forEach(([sym, head, body], i) => {
		const x = M + i * (cw + GAP);
		card(s, x, TOP + 0.1, cw, 2.75);
		text(s, sym, x + 0.3, TOP + 0.3, 0.9, 0.7, { fontSize: 36, bold: true, color: C.accent5, fontFace: THEME.headFontFace });
		text(s, head, x + 1.05, TOP + 0.42, cw - 1.3, 0.5, { fontSize: 17, bold: true, color: C.text2, valign: "middle" });
		text(s, body, x + 0.3, TOP + 1.15, cw - 0.6, 1.6, { fontSize: 15 });
	});
	const y = TOP + 3.2;
	const hw = (CW - GAP) / 2;
	stat(s, M, y, hw, `${fact("model.constraints")} + ${fact("model.warnings")}`,
		"ограничений допустимости и предупреждений: недостижимые сочетания параметров обнаруживаются до генерации данных и исправляются (repair)",
		{ color: C.accent5, h: 1.75 });
	stat(s, M + hw + GAP, y, hw, "YAML / JSON",
		"декларативное описание с JSON Schema; идентификатор сценария — хэш его содержимого, поэтому результаты кэшируются и воспроизводятся",
		{ color: C.accent5, h: 1.75 });
	say(s, "Будаев К. В.",
		`Сценарий в нашей модели — кортеж из четырёх частей: базовая конфигурация потока, список событий дрейфа, протокол оценки и seed реализации. ` +
		`Каждое событие задаётся центром и шириной перехода, формой, видом, величиной и долей затронутых признаков. ` +
		`Над кортежем определены ${fact("model.constraints")} ограничений допустимости и ${fact("model.warnings")} предупреждения: например, величина реального дрейфа ограничена долей затронутых признаков. ` +
		`Недопустимый сценарий обнаруживается до генерации данных. Два seed разделяют геометрию концептов и выборку, поэтому повторы имеют одну и ту же истинную величину дрейфа.`,
	);
}

// =============================================================================
// 6. Единая шкала величины (Будаев К. В.)
// =============================================================================
{
	const s = addSlide("CONTENT_K", SECTIONS[1], "Величина дрейфа — расстояние полной вариации, единое для трёх видов дрейфа");
	const kinds = [
		["реальный: меняется P(y | X)", "m = P( g₀(x) ≠ g₁(x) )", "доля пространства, где меняется метка"],
		["виртуальный: меняется P(X)", "m = 2Φ(δ/2) − 1", "δ — норма сдвига центра признаков"],
		["априорный: меняется P(y)", "m = | π₁ − π₀ |", "изменение доли класса"],
	];
	const cw = (CW - 2 * GAP) / 3;
	kinds.forEach(([head, formula, body], i) => {
		const x = M + i * (cw + GAP);
		card(s, x, TOP, cw, 1.75);
		text(s, head, x + 0.3, TOP + 0.15, cw - 0.6, 0.4, { fontSize: 15, bold: true, color: C.accent5 });
		text(s, formula, x + 0.3, TOP + 0.55, cw - 0.6, 0.55, { fontSize: 22, color: C.text2, fontFace: "Cambria Math", valign: "middle" });
		text(s, body, x + 0.3, TOP + 1.12, cw - 0.6, 0.55, { fontSize: 14, color: C.accent4 });
	});
	const file = FIG("fig_2_1_severity.png");
	const fw = 9.6;
	const fh = pngHeight(file, fw);
	image(s, file, M, TOP + 1.95, fw, fh, "Калибровочная кривая реального дрейфа и наибольшая достижимая величина в зависимости от доли затронутых признаков");
	text(s, "Калибровка аналитическая: угол поворота находится по формуле через двумерное нормальное распределение, без итерационного подбора. Недостижимая величина — нарушение ограничения C10.",
		M + fw + 0.35, TOP + 2.0, CW - fw - 0.35, 3.0, { fontSize: 14 });
	say(s, "Будаев К. В.",
		"Главная идея модели — единая шкала величины. Мы измеряем величину дрейфа расстоянием полной вариации между совместными распределениями до и после события. " +
		"Для калиброванного семейства получены точные формулы: для реального дрейфа это доля пространства, где меняется метка, для виртуального — функция сдвига, для априорного — изменение доли класса. " +
		"Калибровка аналитическая, а не итерационная, как в известных генераторах. На графике видно, что при малой доле затронутых признаков и при дисбалансе большая величина недостижима: " +
		"без проверки генератор молча выдал бы более слабый дрейф, чем заявлен.",
	);
}

// =============================================================================
// 7. Формирование наборов (Будаев К. В.)
// =============================================================================
{
	const s = addSlide("CONTENT_K", SECTIONS[1], "Наборы сценариев формируются автоматически: план Соболя и условные домены");
	const steps = [`Пространство: ${fact("space.free")} параметров`, "План эксперимента: Соболь", "Сценарий по шаблону", "Учёт ограничений", "Набор и манифест"];
	const sw = (CW - 4 * 0.45) / 5;
	steps.forEach((t, i) => {
		const x = M + i * (sw + 0.45);
		s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: TOP, w: sw, h: 0.85, rectRadius: 0.08, fill: { color: C.text2 },
			line: { type: "none" }, objectName: oname("step") });
		text(s, t, x + 0.12, TOP, sw - 0.24, 0.85, { fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle" });
		if (i < steps.length - 1) {
			s.addShape(pres.shapes.RIGHT_ARROW, { x: x + sw + 0.08, y: TOP + 0.3, w: 0.29, h: 0.25, fill: { color: C.accent4 },
				line: { type: "none" }, objectName: oname("arrow") });
		}
	});
	text(s, "Если точка плана недопустима, возможны три способа учёта ограничений:", M, TOP + 1.15, CW, 0.4, { fontSize: 16, bold: true, color: C.text2 });
	const cards = [
		["Отклонение", fact("form.reject_accept"), `точек принято; доля реального дрейфа падает с ${fact("form.real_plan")} до ${fact("form.real_reject")} — баланс плана нарушен`, C.accent4, C.background2],
		["Проекция на границу", fact("form.repair_boundary"), "сценариев оказываются на границе достижимости — набор смещён к «крайним» сценариям", C.accent4, C.background2],
		["Условные домены (наш выбор)", fact("form.adapt_accept"), "точек принято без скопления на границе; допустимость доказана (утверждение 7)", C.accent3, "E3F5EE"],
	];
	const cw = (CW - 2 * GAP) / 3;
	cards.forEach(([head, value, body, color, fill], i) => {
		const x = M + i * (cw + GAP);
		card(s, x, TOP + 1.7, cw, 2.75, fill);
		text(s, head, x + 0.3, TOP + 1.85, cw - 0.6, 0.45, { fontSize: 16, bold: true, color: C.text2 });
		stat(s, x + 0.3, TOP + 2.3, cw - 0.6, value, body, { color, h: 2.0, labelSize: 14 });
	});
	text(s, `Равномерность плана (CD-расхождение, меньше — лучше): Соболь ${fact("form.cd_sobol")}, латинский гиперкуб ${fact("form.cd_lhs")}, случайная выборка ${fact("form.cd_random")}. План Соболя расширяем: набор FULL содержит набор FAST.`,
		M, TOP + 4.65, CW, 0.6, { fontSize: 14, color: C.accent4 });
	say(s, "Будаев К. В.",
		`Метод формирования наборов — центральный результат. Пространство из ${fact("space.free")} параметров покрывается планом эксперимента: ` +
		`последовательность Соболя дала самое равномерное покрытие, ${fact("form.cd_sobol")} против ${fact("form.cd_lhs")} у латинского гиперкуба. ` +
		`Часть точек плана недопустима, и мы сравнили три способа учёта ограничений. Отклонение теряет пятую часть бюджета и смещает баланс видов дрейфа. ` +
		`Проекция ставит ${fact("form.repair_boundary")} сценариев на границу достижимости. Предложенные условные домены сужают домен зависимого параметра и сохраняют ${fact("form.adapt_accept")} точек без скопления на границе — этот способ выбран по умолчанию.`,
	);
}

// =============================================================================
// 8. Проверка генератора (Будаев К. В.)
// =============================================================================
{
	const s = addSlide("CONTENT_K", SECTIONS[1], "Генератор воспроизводит заданную величину дрейфа в пределах выборочной погрешности");
	const file = FIG("fig_5_2_e1_zscores.png");
	const fw = 7.0;
	const fh = pngHeight(file, fw);
	image(s, file, M, TOP + 0.2, fw, fh, "Распределение нормированных отклонений фактической величины от заданной и плотность N(0, 1)");
	text(s, `Нормированные отклонения z = (m̂ − m)/SE на ${fact("e1.rand_n")} случайных сценариях; кривая — N(0, 1)`,
		M, TOP + 0.3 + fh, fw, 0.6, { fontSize: 13, color: C.accent4 });
	const x = M + fw + 0.5;
	const w = CW - fw - 0.5;
	stat(s, x, TOP + 0.1, w, `${fact("e1.rand_sd_min")} – ${fact("e1.rand_sd_max")}`,
		"стандартное отклонение z для трёх видов дрейфа (ожидается 1); смещения не обнаружено", { color: C.accent5, h: 1.55 });
	stat(s, x, TOP + 1.75, w, `≤ ${fact("e1.shape_max")}`,
		"отклонение эмпирической кривой перехода от теоретической для четырёх форм", { color: C.accent5, h: 1.4 });
	stat(s, x, TOP + 3.25, w, `≤ ${fact("e1.real_err")}`,
		"ошибка величины при внедрении дрейфа в реальные данные Bananas и Phishing", { color: C.accent5, h: 1.4 });
	say(s, "Будаев К. В.",
		`Эксперимент Э1 проверяет генератор по данным, независимо от истинной разметки. На ${fact("e1.rand_n")} случайных сценариях отклонения фактической величины от заданной ` +
		`ведут себя как стандартный нормальный закон: стандартное отклонение от ${fact("e1.rand_sd_min")} до ${fact("e1.rand_sd_max")}, систематического смещения нет. ` +
		`Формы перехода совпадают с теорией, а на реальных данных величина воспроизводится точно до массы одной строки. ` +
		`Генератор надёжен, и на нём можно сравнивать детекторы. Об этом расскажет Тимур Гарифзянов.`,
	);
}

// =============================================================================
// 9. Фреймворк (Гарифзянов Т. Р.)
// =============================================================================
pres.addSection({ title: SECTIONS[2] });
{
	const s = addSlide("CONTENT_T", SECTIONS[2], "scendrift: от описания сценария до статистического сравнения детекторов");
	const file = FIG("arch_dataflow.png");
	const fw = 8.9;
	const fh = pngHeight(file, fw);
	image(s, file, M, TOP + 0.05, fw, fh, "Конвейер: пространство параметров, план, сборка, проверка, набор, генерация, раннер, метрики, анализ, отчёт");
	const x = M + fw + 0.45;
	const w = CW - fw - 0.45;
	const stats = [
		[fact("pkg.modules"), "модули пакета Python"],
		[fact("pkg.lines"), "строки исходного кода"],
		[fact("tests.count"), "модульные тесты pytest"],
		["9 + 3", "детекторы river и базовые линии NoDrift, Periodic, Oracle"],
	];
	stats.forEach(([v, l], i) => stat(s, x, TOP + i * 1.2, w, v, l, { color: C.accent6, h: 1.15, valueSize: 30, labelSize: 13 }));
	text(s, "Протокол prequential: модель предсказывает, затем обучается; детектор получает поток ошибок; срабатывание сопоставляется с истинной разметкой через окно допуска",
		M, TOP + 0.2 + fh, fw, 0.8, { fontSize: 14, color: C.accent4 });
	say(s, "Гарифзянов Т. Р.",
		`Модель и метод реализованы во фреймворке scendrift. Верхняя строка конвейера — формирование набора сценариев, нижняя — исполнение и анализ. ` +
		`Генератор строит поток с истинной разметкой, раннер прогоняет девять детекторов river и три базовые линии по схеме prequential, ` +
		`затем вычисляются метрики обнаружения и выполняется статистическое сравнение по Демшару. ` +
		`При проверке адаптеров мы нашли расхождение описания и реализации FHDDM в river и исправили вход. Корректность закрывают модульные тесты, всего ${fact("tests.count")}.`,
	);
}

// =============================================================================
// 10. Э2 (Гарифзянов Т. Р.)
// =============================================================================
{
	const s = addSlide("CONTENT_T", SECTIONS[2], `На равномерно покрывающем наборе средний F1 детекторов — от ${fact("e2.f1_min")} до ${fact("e2.f1_max")}`);
	const labels = [...S.order, "Periodic(2000)"];
	const values = [...S.f1, S.f1_periodic];
	const colors = labels.map((l) => (l.startsWith("Periodic") ? HEX.accent2 : HEX.accent1));
	s.addChart(pres.charts.BAR, [{ name: "средний F1", labels, values }], chartStyle({
		x: M, y: TOP, w: 7.9, h: 5.05, barDir: "bar", catAxisOrientation: "maxMin", chartColors: colors,
		valAxisMinVal: 0, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0", dataLabelFormatCode: "0.00",
		showTitle: true, title: `Средний F1 на наборе Э2 (протокол reset)`, showLegend: false, barGapWidthPct: 45,
		objectName: oname("chart"),
	}));
	const x = M + 7.9 + 0.45;
	const w = CW - 7.9 - 0.45;
	card(s, x, TOP + 0.1, w, 2.55, "FDEDE6");
	stat(s, x + 0.3, TOP + 0.25, w - 0.6, fact("e2.f1_periodic"),
		"F1 периодического сброса модели каждые 2000 объектов — без анализа данных он выше любого детектора", { color: C.accent2, h: 2.25 });
	stat(s, x, TOP + 2.95, w, fact("e2.runs"),
		`прогонов: ${fact("e2.scenarios")} сценариев набора ${fact("suite.name")}, повторов — ${fact("e2.repeats")}, методов — ${fact("e2.detectors")}`,
		{ color: C.accent6, h: 1.4, valueSize: 30 });
	say(s, "Гарифзянов Т. Р.",
		`Основной бенчмарк Э2 выполнен на автоматически сформированном наборе из ${fact("e2.scenarios")} сценариев. ` +
		`Качество обнаружения низкое и неоднородное: средний F1 детекторов от ${fact("e2.f1_min")} у ${fact("e2.f1_min_det")} до ${fact("e2.f1_max")} у ${fact("e2.f1_max_det")}. ` +
		`Различия значимы по критерию Фридмана. Но периодический сброс, который вообще не смотрит на данные, получает F1 ${fact("e2.f1_periodic")} — выше любого детектора: ` +
		`его срабатывания просто попадают в окна допуска. Это известная «иллюзия прогресса», и мы показали её количественно.`,
	);
}

// =============================================================================
// 11. Иллюзия прогресса (Гарифзянов Т. Р.)
// =============================================================================
{
	const s = addSlide("CONTENT_T", SECTIONS[2], "F1 и польза для модели расходятся: F1 нельзя использовать как единственную метрику");
	const w = 4.2;
	stat(s, M, TOP + 0.05, w, `${fact("e5.worse_periodic")} из 9`,
		"детекторов значимо хуже периодического сброса по F1 (Уилкоксон, поправка Холма)", { color: C.accent2, h: 1.6 });
	stat(s, M, TOP + 1.75, w, `${fact("e5.acc_better")} из 9`,
		"детекторов значимо лучше периодического сброса по приросту accuracy", { color: C.accent6, h: 1.45 });
	stat(s, M, TOP + 3.3, w, `τ = ${fact("e8.tau")}`,
		"согласие ранжирований детекторов по F1 и по приросту accuracy — связи нет", { color: C.text2, h: 1.6 });
	const labels = [...S.order, "Periodic(2000)"];
	const values = [...S.dacc, S.dacc_periodic];
	const colors = labels.map((l) => (l.startsWith("Periodic") ? HEX.accent2 : HEX.accent6));
	s.addChart(pres.charts.BAR, [{ name: "Δaccuracy", labels, values }], chartStyle({
		x: M + w + 0.45, y: TOP, w: CW - w - 0.45, h: 5.05, barDir: "bar", catAxisOrientation: "maxMin",
		chartColors: colors, invertedColors: colors, valAxisLabelFormatCode: "0.00", dataLabelFormatCode: "0.000",
		valAxisMinVal: -0.02, valAxisMaxVal: 0.02, valAxisMajorUnit: 0.01, catAxisLabelPos: "low",
		showTitle: true, title: "Средний прирост accuracy относительно модели без детектора", showLegend: false, barGapWidthPct: 45,
		objectName: oname("chart"),
	}));
	say(s, "Гарифзянов Т. Р.",
		`Статистическое сравнение Э5 подтверждает это. По F1 ${fact("e5.worse_periodic")} из девяти детекторов значимо хуже периодического сброса, ` +
		`а по приросту accuracy базовой модели, наоборот, все ${fact("e5.acc_better")} лучше него — периодический сброс здесь худший. ` +
		`Ранжирования по F1 и по accuracy не связаны: τ Кендалла равно ${fact("e8.tau")}. Порядок детекторов на диаграмме тот же, что на предыдущем слайде. ` +
		`Вывод для практики: F1 с окнами допуска нужно дополнять периодической базовой линией, частотой ложных тревог и сравнением со случайным уровнем.`,
	);
}

// =============================================================================
// 12. Что определяет сложность (Гарифзянов Т. Р.)
// =============================================================================
{
	const s = addSlide("CONTENT_T", SECTIONS[2], "Сложность определяет величина дрейфа, а виртуальный дрейф по потоку ошибок не виден");
	const w = 4.2;
	stat(s, M, TOP + 0.05, w, `${fact("e3.rho_m_min")} – ${fact("e3.rho_m_max")}`,
		"корреляция F1 с величиной дрейфа на сетке «величина × ширина»", { color: C.accent6, h: 1.5 });
	stat(s, M, TOP + 1.65, w, `≤ ${fact("e3.rho_w_max")}`,
		"модуль корреляции F1 с шириной перехода — связь слабая", { color: C.accent6, h: 1.45 });
	stat(s, M, TOP + 3.2, w, "0",
		"значимых связей F1 с долей затронутых признаков при фиксированной величине — шкала её уже учитывает", { color: C.accent6, h: 1.75 });
	s.addChart(pres.charts.BAR, [
		{ name: "реальный дрейф", labels: S.order, values: S.excess_real },
		{ name: "виртуальный дрейф", labels: S.order, values: S.excess_virtual },
	], chartStyle({
		x: M + w + 0.45, y: TOP, w: CW - w - 0.45, h: 5.05, barDir: "bar", catAxisOrientation: "maxMin",
		chartColors: [HEX.accent1, HEX.accent2], valAxisMinVal: -0.4, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.2,
		valAxisLabelFormatCode: "0.0", dataLabelFormatCode: "0.00", catAxisLabelPos: "low",
		showTitle: true, title: "Превышение recall над случайным уровнем (по частоте ложных тревог)",
		showLegend: true, legendPos: "b", barGapWidthPct: 40, objectName: oname("chart"),
	}));
	say(s, "Гарифзянов Т. Р.",
		`Что определяет сложность обнаружения? На контролируемой сетке F1 растёт с величиной дрейфа: корреляция от ${fact("e3.rho_m_min")} до ${fact("e3.rho_m_max")}, ` +
		`а с шириной перехода связь слабая. Доля затронутых признаков при фиксированной величине не влияет — это подтверждает единую шкалу. ` +
		`На диаграмме — превышение recall над случайным уровнем, вычисленным по частоте ложных тревог. На реальном дрейфе оно доходит до ${fact("e3.real_excess_max")}, ` +
		`а на виртуальном не больше ${fact("e3.virt_excess_max")}: по потоку ошибок виртуальный дрейф не виден, что следует из модели. У EDDM обнаружения на уровне случайных.`,
	);
}

// =============================================================================
// 13. Ручной набор (Гарифзянов Т. Р.)
// =============================================================================
{
	const s = addSlide("CONTENT_T", SECTIONS[2], "Ручной набор классических потоков завышает качество и меняет порядок детекторов");
	const cwid = 7.9;
	s.addChart(pres.charts.BAR, [
		{ name: "ручной набор (SEA, Agrawal, STAGGER, Sine, Mixed)", labels: S.order, values: S.f1_manual },
		{ name: `автоматический набор ${fact("suite.name")}`, labels: S.order, values: S.f1_auto },
	], chartStyle({
		x: M, y: TOP, w: cwid, h: 5.05, barDir: "bar", catAxisOrientation: "maxMin",
		chartColors: ["A7B1BF", HEX.accent6], valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2,
		valAxisLabelFormatCode: "0.0", dataLabelFormatCode: "0.00",
		showTitle: true, title: "Средний F1 детекторов", showLegend: true, legendPos: "b", barGapWidthPct: 40,
		objectName: oname("chart"),
	}));
	const x = M + cwid + 0.45;
	const w = CW - cwid - 0.45;
	stat(s, x, TOP + 0.05, w, `τ = ${fact("e6.tau")}`,
		`${fact("e6.p")}: согласие ранжирований ручного и автоматического наборов не подтверждено`, { color: C.accent6, h: 1.5 });
	stat(s, x, TOP + 1.65, w, `${fact("e6.classes_manual")} и ${fact("e6.classes_auto")}`,
		"классов таксономии покрывают ручной и автоматический наборы", { color: C.accent6, h: 1.4 });
	stat(s, x, TOP + 3.15, w, `+${fact("e7.gap_virtual")}`,
		"recall на виртуальном дрейфе на реальных данных по сравнению с синтетикой (Э7): перенос выводов частичный", { color: C.accent6, h: 1.8 });
	say(s, "Гарифзянов Т. Р.",
		`Совпадают ли выводы с классическим ручным набором? Нет. На ручном наборе средний F1 от ${fact("e6.f1_min")} до ${fact("e6.f1_max")}, ` +
		`на автоматическом — от ${fact("e2.f1_min")} до ${fact("e2.f1_max")}. Согласие ранжирований не подтверждено, τ равно ${fact("e6.tau")}. ` +
		`Ручной набор покрывает ${fact("e6.classes_manual")} класса таксономии и в основном большие изменения, автоматический — ${fact("e6.classes_auto")}. ` +
		`На реальных данных с внедрённым дрейфом виртуальный дрейф виден лучше, потому что модель там несовершенна. Это количественный аргумент за автоматическое формирование наборов.`,
	);
}

// =============================================================================
// 14. Новизна и значимость (Гарифзянов Т. Р.)
// =============================================================================
pres.addSection({ title: SECTIONS[3] });
{
	const s = addSlide("CONTENT_T", SECTIONS[3], "Научная новизна и практическая значимость");
	const items = [
		"Формальная модель сценария дрейфа с проверяемыми ограничениями допустимости и разделением структурной случайности и случайности реализации",
		"Единая шкала величины дрейфа — расстояние полной вариации — с аналитической калибровкой для реального, виртуального и априорного дрейфа",
		"Метод формирования наборов сценариев по планам вычислительного эксперимента с условными доменами",
		"Методика описания областей отказа детекторов с поправкой на случайный уровень обнаружения",
	];
	const lw = 7.6;
	items.forEach((t, i) => {
		const y = TOP + i * 1.22;
		card(s, M, y, lw, 1.08);
		circleNumber(s, i + 1, M + 0.25, y + 0.29, C.accent5);
		text(s, t, M + 0.95, y + 0.1, lw - 1.15, 0.9, { fontSize: 15, valign: "middle" });
	});
	const x = M + lw + 0.45;
	const w = CW - lw - 0.45;
	card(s, x, TOP, w, 4.74, C.text2);
	text(s, "Практическая значимость", x + 0.35, TOP + 0.25, w - 0.7, 0.5, { fontSize: 18, bold: true, color: C.background1 });
	bullets(s, [
		"выбор детектора под величину и скорость дрейфа, дисбаланс и шум конкретной задачи",
		"проверка новых детекторов на воспроизводимых наборах сценариев",
		"протокол оценки без «иллюзии прогресса»: базовые линии и случайный уровень",
	], x + 0.35, TOP + 0.9, w - 0.7, 3.6, { fontSize: 15, color: C.background1, paraSpaceAfter: 12 });
	say(s, "Гарифзянов Т. Р.",
		"Научная новизна работы — в четырёх результатах: формальная модель сценария с проверяемыми ограничениями, " +
		"единая аналитически калиброванная шкала величины для трёх видов дрейфа, метод формирования наборов с условными доменами " +
		"и методика описания областей отказа с поправкой на случайный уровень. Практически фреймворк позволяет выбирать детектор под конкретную задачу " +
		"и проверять новые детекторы на воспроизводимых наборах. Детекторы, генераторы классических потоков и статистические критерии заимствованы, это указано в работе.",
	);
}

// =============================================================================
// 15. Выводы (Гарифзянов Т. Р.)
// =============================================================================
{
	const s = addSlide("DARK_T", SECTIONS[3], "Выводы");
	const items = [
		"Сценарий дрейфа формализован, его допустимость проверяется до генерации данных",
		"Единая шкала величины реализована и подтверждена: генератор точен в пределах выборочной погрешности",
		"Наборы формируются автоматически; условные домены сохраняют бюджет и баланс плана",
		`Средний F1 детекторов river на равномерном наборе — от ${fact("e2.f1_min")} до ${fact("e2.f1_max")}; F1 нужно дополнять базовыми линиями`,
		"Главный фактор сложности — величина дрейфа; ручной набор завышает качество и даёт другой порядок",
	];
	items.forEach((t, i) => {
		const y = TOP + i * 0.8;
		circleNumber(s, i + 1, M, y + 0.06, C.accent3, 0.46);
		text(s, t, M + 0.7, y, CW - 0.7, 0.62, { fontSize: 17, color: C.background1, valign: "middle" });
	});
	const note = fact("run.mode") === "FULL"
		? "Ограничения: одно калиброванное синтетическое семейство, одна базовая модель, параметры детекторов по умолчанию."
		: "Ограничения: одно калиброванное синтетическое семейство, одна базовая модель, параметры детекторов по умолчанию; числа получены в режиме FAST.";
	text(s, note, M, TOP + 4.0, CW, 0.65, { fontSize: 14, color: C.background2 });
	text(s, "Спасибо за внимание! Готовы ответить на вопросы.", M, TOP + 4.75, CW, 0.4, { fontSize: 16, bold: true, color: C.background1 });
	say(s, "Гарифзянов Т. Р.",
		"Подведём итоги. Цель достигнута: сценарий дрейфа формализован и проверяется до генерации, единая шкала величины реализована и подтверждена, " +
		"наборы сценариев формируются автоматически. Эксперименты показали, что на равномерном наборе качество детекторов низкое, F1 нужно дополнять базовыми линиями, " +
		"главный фактор сложности — величина дрейфа, а ручные бенчмарки завышают качество. Развитие работы — нелинейные калиброванные семейства и детекторы без учителя. " +
		"Спасибо за внимание, мы готовы ответить на вопросы.",
	);
}

// --- запись и тема ----------------------------------------------------------

// pptxgenjs записывает в тему стандартную палитру Office; заменяем её своей,
// чтобы цвета схемы (SchemeColor) соответствовали THEME.
async function writeTheme(file) {
	const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
	const zip = await JSZip.loadAsync(fs.readFileSync(file));
	const part = "ppt/theme/theme1.xml";
	const slots = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
	const scheme = `<a:clrScheme name="${THEME.name}">` +
		slots.map((k) => `<a:${k}><a:srgbClr val="${THEME.colors[k]}"/></a:${k}>`).join("") + "</a:clrScheme>";
	const xml = (await zip.file(part).async("string"))
		.replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, () => scheme)
		.replace(/(<a:(?:theme|fontScheme)\b[^>]*?\bname=")[^"]*"/g, (_, head) => `${head}${THEME.name}"`);
	if (!xml.includes(scheme)) throw new Error("в теме нет <a:clrScheme>");
	zip.file(part, xml);
	for (const name of Object.keys(zip.files)) {
		if (!name.endsWith(".xml")) continue;
		const bad = (await zip.file(name).async("string")).match(/<a:srgbClr val="((?![0-9A-Fa-f]{6}")[^"]*)"/);
		if (bad) throw new Error(`${name}: цвет схемы в параметре, который принимает только hex (${bad[1]})`);
	}
	fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
	await pres.writeFile({ fileName: OUT });
	await writeTheme(OUT);
	const total = {};
	for (const { who, seconds } of timing) total[who] = (total[who] || 0) + seconds;
	const all = Object.values(total).reduce((x, y) => x + y, 0);
	const fmt = (sec) => `${Math.floor(sec / 60)} мин ${sec % 60} с`;
	console.log(`${path.relative(ROOT, OUT)}: слайдов ${pres.slides.length}; доклад ≈ ${fmt(all)} (` +
		Object.entries(total).map(([who, sec]) => `${who} — ${fmt(sec)}`).join(", ") + ")");
	if (timing.length !== pres.slides.length) throw new Error("не у всех слайдов есть заметки докладчика");
})().catch((e) => {
	console.error(e);
	process.exit(1);
});
