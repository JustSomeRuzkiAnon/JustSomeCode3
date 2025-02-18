import { Request, RequestHandler, Router } from "express";
import { v4 } from "uuid";
import { GoogleAIKey, keyPool } from "../shared/key-management";
import { config } from "../config";
import { ipLimiter } from "./rate-limit";
import {
  createPreprocessorMiddleware,
  finalizeSignedRequest,
} from "./middleware/request";
import { ProxyResHandlerWithBody } from "./middleware/response";
import { addGoogleAIKey } from "./middleware/request/mutators/add-google-ai-key";
import { createQueuedProxyMiddleware } from "./middleware/request/proxy-middleware-factory";

let modelsCache: any = null;
let modelsCacheTime = 0;

// https://ai.google.dev/models/gemini
// TODO: list models https://ai.google.dev/tutorials/rest_quickstart#list_models

const getModelsResponse = () => {
  if (new Date().getTime() - modelsCacheTime < 1000 * 60) {
    return modelsCache;
  }

  if (!config.googleAIKey) return { object: "list", data: [] };

  const keys = keyPool
    .list()
    .filter((k) => k.service === "google-ai") as GoogleAIKey[];
  if (keys.length === 0) {
    modelsCache = { object: "list", data: [] };
    modelsCacheTime = new Date().getTime();
    return modelsCache;
  }

  const modelIds = Array.from(
    new Set(keys.map((k) => k.modelIds).flat())
  ).filter((id) => id.startsWith("models/gemini"));
  const models = modelIds.map((id) => ({
    id,
    object: "model",
    created: new Date().getTime(),
    owned_by: "google",
    permission: [],
    root: "google",
    parent: null,
  }));

  modelsCache = { object: "list", data: models };
  modelsCacheTime = new Date().getTime();

  return modelsCache;
};

const handleModelRequest: RequestHandler = (_req, res) => {
  res.status(200).json(getModelsResponse());
};

// Native Gemini API model list request. Ideally shouldn't be hardcoded
// Last updated: Feb 5, 2025
const handleNativeModelRequest: RequestHandler = (_req, res) => {
  res.status(200).json({
    "models":[{"name":"models/chat-bison-001","version":"001","displayName":"PaLM 2 Chat (Legacy)","description":"A legacy text-only model optimized for chat conversations","inputTokenLimit":4096,"outputTokenLimit":1024,"supportedGenerationMethods":["generateMessage","countMessageTokens"],"temperature":0.25,"topP":0.95,"topK":40},{"name":"models/text-bison-001","version":"001","displayName":"PaLM 2 (Legacy)","description":"A legacy model that understands text and generates text as an output","inputTokenLimit":8196,"outputTokenLimit":1024,"supportedGenerationMethods":["generateText","countTextTokens","createTunedTextModel"],"temperature":0.7,"topP":0.95,"topK":40},{"name":"models/embedding-gecko-001","version":"001","displayName":"Embedding Gecko","description":"Obtain a distributed representation of a text.","inputTokenLimit":1024,"outputTokenLimit":1,"supportedGenerationMethods":["embedText","countTextTokens"]},{"name":"models/gemini-1.0-pro-latest","version":"001","displayName":"Gemini 1.0 Pro Latest","description":"The original Gemini 1.0 Pro model. This model will be discontinued on February 15th, 2025. Move to a newer Gemini version.","inputTokenLimit":30720,"outputTokenLimit":2048,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":0.9,"topP":1},{"name":"models/gemini-1.0-pro","version":"001","displayName":"Gemini 1.0 Pro","description":"The best model for scaling across a wide range of tasks","inputTokenLimit":30720,"outputTokenLimit":2048,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":0.9,"topP":1},{"name":"models/gemini-pro","version":"001","displayName":"Gemini 1.0 Pro","description":"The best model for scaling across a wide range of tasks","inputTokenLimit":30720,"outputTokenLimit":2048,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":0.9,"topP":1},{"name":"models/gemini-1.0-pro-001","version":"001","displayName":"Gemini 1.0 Pro 001 (Tuning)","description":"The original Gemini 1.0 Pro model version that supports tuning. Gemini 1.0 Pro will be discontinued on February 15th, 2025. Move to a newer Gemini version.","inputTokenLimit":30720,"outputTokenLimit":2048,"supportedGenerationMethods":["generateContent","countTokens","createTunedModel"],"temperature":0.9,"topP":1},{"name":"models/gemini-1.0-pro-vision-latest","version":"001","displayName":"Gemini 1.0 Pro Vision","description":"The original Gemini 1.0 Pro Vision model version which was optimized for image understanding. Gemini 1.0 Pro Vision was deprecated on July 12, 2024. Move to a newer Gemini version.","inputTokenLimit":12288,"outputTokenLimit":4096,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":0.4,"topP":1,"topK":32},{"name":"models/gemini-pro-vision","version":"001","displayName":"Gemini 1.0 Pro Vision","description":"The original Gemini 1.0 Pro Vision model version which was optimized for image understanding. Gemini 1.0 Pro Vision was deprecated on July 12, 2024. Move to a newer Gemini version.","inputTokenLimit":12288,"outputTokenLimit":4096,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":0.4,"topP":1,"topK":32},{"name":"models/gemini-1.5-pro-latest","version":"001","displayName":"Gemini 1.5 Pro Latest","description":"Alias that points to the most recent production (non-experimental) release of Gemini 1.5 Pro, our mid-size multimodal model that supports up to 2 million tokens.","inputTokenLimit":2000000,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-1.5-pro-001","version":"001","displayName":"Gemini 1.5 Pro 001","description":"Stable version of Gemini 1.5 Pro, our mid-size multimodal model that supports up to 2 million tokens, released in May of 2024.","inputTokenLimit":2000000,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens","createCachedContent"],"temperature":1,"topP":0.95,"topK":64,"maxTemperature":2},{"name":"models/gemini-1.5-pro-002","version":"002","displayName":"Gemini 1.5 Pro 002","description":"Stable version of Gemini 1.5 Pro, our mid-size multimodal model that supports up to 2 million tokens, released in September of 2024.","inputTokenLimit":2000000,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens","createCachedContent"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-1.5-pro","version":"001","displayName":"Gemini 1.5 Pro","description":"Stable version of Gemini 1.5 Pro, our mid-size multimodal model that supports up to 2 million tokens, released in May of 2024.","inputTokenLimit":2000000,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-1.5-flash-latest","version":"001","displayName":"Gemini 1.5 Flash Latest","description":"Alias that points to the most recent production (non-experimental) release of Gemini 1.5 Flash, our fast and versatile multimodal model for scaling across diverse tasks.","inputTokenLimit":1000000,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-1.5-flash-001","version":"001","displayName":"Gemini 1.5 Flash 001","description":"Stable version of Gemini 1.5 Flash, our fast and versatile multimodal model for scaling across diverse tasks, released in May of 2024.","inputTokenLimit":1000000,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens","createCachedContent"],"temperature":1,"topP":0.95,"topK":64,"maxTemperature":2},{"name":"models/gemini-1.5-flash-001-tuning","version":"001","displayName":"Gemini 1.5 Flash 001 Tuning","description":"Version of Gemini 1.5 Flash that supports tuning, our fast and versatile multimodal model for scaling across diverse tasks, released in May of 2024.","inputTokenLimit":16384,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens","createTunedModel"],"temperature":1,"topP":0.95,"topK":64,"maxTemperature":2},{"name":"models/gemini-1.5-flash","version":"001","displayName":"Gemini 1.5 Flash","description":"Alias that points to the most recent stable version of Gemini 1.5 Flash, our fast and versatile multimodal model for scaling across diverse tasks.","inputTokenLimit":1000000,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-1.5-flash-002","version":"002","displayName":"Gemini 1.5 Flash 002","description":"Stable version of Gemini 1.5 Flash, our fast and versatile multimodal model for scaling across diverse tasks, released in September of 2024.","inputTokenLimit":1000000,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens","createCachedContent"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-1.5-flash-8b","version":"001","displayName":"Gemini 1.5 Flash-8B","description":"Stable version of Gemini 1.5 Flash-8B, our smallest and most cost effective Flash model, released in October of 2024.","inputTokenLimit":1000000,"outputTokenLimit":8192,"supportedGenerationMethods":["createCachedContent","generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-1.5-flash-8b-001","version":"001","displayName":"Gemini 1.5 Flash-8B 001","description":"Stable version of Gemini 1.5 Flash-8B, our smallest and most cost effective Flash model, released in October of 2024.","inputTokenLimit":1000000,"outputTokenLimit":8192,"supportedGenerationMethods":["createCachedContent","generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-1.5-flash-8b-latest","version":"001","displayName":"Gemini 1.5 Flash-8B Latest","description":"Alias that points to the most recent production (non-experimental) release of Gemini 1.5 Flash-8B, our smallest and most cost effective Flash model, released in October of 2024.","inputTokenLimit":1000000,"outputTokenLimit":8192,"supportedGenerationMethods":["createCachedContent","generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-1.5-flash-8b-exp-0827","version":"001","displayName":"Gemini 1.5 Flash 8B Experimental 0827","description":"Experimental release (August 27th, 2024) of Gemini 1.5 Flash-8B, our smallest and most cost effective Flash model. Replaced by Gemini-1.5-flash-8b-001 (stable).","inputTokenLimit":1000000,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-1.5-flash-8b-exp-0924","version":"001","displayName":"Gemini 1.5 Flash 8B Experimental 0924","description":"Experimental release (September 24th, 2024) of Gemini 1.5 Flash-8B, our smallest and most cost effective Flash model. Replaced by Gemini-1.5-flash-8b-001 (stable).","inputTokenLimit":1000000,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-2.0-flash-exp","version":"2.0","displayName":"Gemini 2.0 Flash Experimental","description":"Gemini 2.0 Flash Experimental","inputTokenLimit":1048576,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens","bidiGenerateContent"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-2.0-flash","version":"2.0","displayName":"Gemini 2.0 Flash","description":"Gemini 2.0 Flash","inputTokenLimit":1048576,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens","bidiGenerateContent"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-2.0-flash-001","version":"2.0","displayName":"Gemini 2.0 Flash 001","description":"Stable version of Gemini 2.0 Flash, our fast and versatile multimodal model for scaling across diverse tasks, released in January of 2025.","inputTokenLimit":1048576,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens","bidiGenerateContent"],"temperature":1,"topP":0.95,"topK":40,"maxTemperature":2},{"name":"models/gemini-2.0-flash-lite-preview","version":"preview-02-05","displayName":"Gemini 2.0 Flash-Lite Preview","description":"Preview release (February 5th, 2025) of Gemini 2.0 Flash Lite","inputTokenLimit":1048576,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":64,"maxTemperature":2},{"name":"models/gemini-2.0-flash-lite-preview-02-05","version":"preview-02-05","displayName":"Gemini 2.0 Flash-Lite Preview 02-05","description":"Preview release (February 5th, 2025) of Gemini 2.0 Flash Lite","inputTokenLimit":1048576,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":64,"maxTemperature":2},{"name":"models/gemini-2.0-pro-exp","version":"2.0","displayName":"Gemini 2.0 Pro Experimental","description":"Experimental release (February 5th, 2025) of Gemini 2.0 Pro","inputTokenLimit":2097152,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":64,"maxTemperature":2},{"name":"models/gemini-2.0-pro-exp-02-05","version":"2.0","displayName":"Gemini 2.0 Pro Experimental 02-05","description":"Experimental release (February 5th, 2025) of Gemini 2.0 Pro","inputTokenLimit":2097152,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":64,"maxTemperature":2},{"name":"models/gemini-exp-1206","version":"2.0","displayName":"Gemini 2.0 Pro Experimental","description":"Experimental release (February 5th, 2025) of Gemini 2.0 Pro","inputTokenLimit":2097152,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":64,"maxTemperature":2},{"name":"models/gemini-2.0-flash-thinking-exp-01-21","version":"2.0-exp-01-21","displayName":"Gemini 2.0 Flash Thinking Experimental 01-21","description":"Experimental release (January 21st, 2025) of Gemini 2.0 Flash Thinking","inputTokenLimit":1048576,"outputTokenLimit":65536,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":0.7,"topP":0.95,"topK":64,"maxTemperature":2},{"name":"models/gemini-2.0-flash-thinking-exp","version":"2.0-exp-01-21","displayName":"Gemini 2.0 Flash Thinking Experimental 01-21","description":"Experimental release (January 21st, 2025) of Gemini 2.0 Flash Thinking","inputTokenLimit":1048576,"outputTokenLimit":65536,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":0.7,"topP":0.95,"topK":64,"maxTemperature":2},{"name":"models/gemini-2.0-flash-thinking-exp-1219","version":"2.0","displayName":"Gemini 2.0 Flash Thinking Experimental","description":"Gemini 2.0 Flash Thinking Experimental","inputTokenLimit":1048576,"outputTokenLimit":65536,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":0.7,"topP":0.95,"topK":64,"maxTemperature":2},{"name":"models/learnlm-1.5-pro-experimental","version":"001","displayName":"LearnLM 1.5 Pro Experimental","description":"Alias that points to the most recent stable version of Gemini 1.5 Pro, our mid-size multimodal model that supports up to 2 million tokens.","inputTokenLimit":32767,"outputTokenLimit":8192,"supportedGenerationMethods":["generateContent","countTokens"],"temperature":1,"topP":0.95,"topK":64,"maxTemperature":2},{"name":"models/embedding-001","version":"001","displayName":"Embedding 001","description":"Obtain a distributed representation of a text.","inputTokenLimit":2048,"outputTokenLimit":1,"supportedGenerationMethods":["embedContent"]},{"name":"models/text-embedding-004","version":"004","displayName":"Text Embedding 004","description":"Obtain a distributed representation of a text.","inputTokenLimit":2048,"outputTokenLimit":1,"supportedGenerationMethods":["embedContent"]},{"name":"models/aqa","version":"001","displayName":"Model that performs Attributed Question Answering.","description":"Model trained to return answers to questions that are grounded in provided sources, along with estimating answerable probability.","inputTokenLimit":7168,"outputTokenLimit":1024,"supportedGenerationMethods":["generateAnswer"],"temperature":0.2,"topP":1,"topK":40}]
  });
};

const googleAIBlockingResponseHandler: ProxyResHandlerWithBody = async (
  _proxyRes,
  req,
  res,
  body
) => {
  if (typeof body !== "object") {
    throw new Error("Expected body to be an object");
  }

  let newBody = body;
  if (req.inboundApi === "openai") {
    req.log.info("Transforming Google AI response to OpenAI format");
    newBody = transformGoogleAIResponse(body, req);
  }

  res.status(200).json({ ...newBody, proxy: body.proxy });
};

function transformGoogleAIResponse(
  resBody: Record<string, any>,
  req: Request
): Record<string, any> {
  const totalTokens = (req.promptTokens ?? 0) + (req.outputTokens ?? 0);
  const parts = resBody.candidates[0].content?.parts ?? [{ text: "" }];
  const content = parts[0].text.replace(/^(.{0,50}?): /, () => "");
  return {
    id: "goo-" + v4(),
    object: "chat.completion",
    created: Date.now(),
    model: req.body.model,
    usage: {
      prompt_tokens: req.promptTokens,
      completion_tokens: req.outputTokens,
      total_tokens: totalTokens,
    },
    choices: [
      {
        message: { role: "assistant", content },
        finish_reason: resBody.candidates[0].finishReason,
        index: 0,
      },
    ],
  };
}

const googleAIProxy = createQueuedProxyMiddleware({
  target: ({ signedRequest }) => {
    if (!signedRequest) throw new Error("Must sign request before proxying");
    const { protocol, hostname} = signedRequest;
    return `${protocol}//${hostname}`;
  },
  mutations: [addGoogleAIKey, finalizeSignedRequest],
  blockingResponseHandler: googleAIBlockingResponseHandler,
});

const googleAIRouter = Router();
googleAIRouter.get("/v1/models", handleModelRequest);
googleAIRouter.get("/:apiVersion(v1alpha|v1beta)/models", handleNativeModelRequest);

// Native Google AI chat completion endpoint
googleAIRouter.post(
  "/:apiVersion(v1alpha|v1beta)/models/:modelId:(generateContent|streamGenerateContent)",
  ipLimiter,
  createPreprocessorMiddleware(
    { inApi: "google-ai", outApi: "google-ai", service: "google-ai" },
    { beforeTransform: [maybeReassignModel], afterTransform: [setStreamFlag] }
  ),
  googleAIProxy
);

// OpenAI-to-Google AI compatibility endpoint.
googleAIRouter.post(
  "/v1/chat/completions",
  ipLimiter,
  createPreprocessorMiddleware(
    { inApi: "openai", outApi: "google-ai", service: "google-ai" },
    { afterTransform: [maybeReassignModel] }
  ),
  googleAIProxy
);

function setStreamFlag(req: Request) {
  const isStreaming = req.url.includes("streamGenerateContent");
  if (isStreaming) {
    req.body.stream = true;
    req.isStreaming = true;
  } else {
    req.body.stream = false;
    req.isStreaming = false;
  }
}

/**
 * Replaces requests for non-Google AI models with gemini-1.5-pro-latest.
 * Also strips models/ from the beginning of the model IDs.
 **/
function maybeReassignModel(req: Request) {
  // Ensure model is on body as a lot of middleware will expect it.
  const model = req.body.model || req.url.split("/").pop()?.split(":").shift();
  if (!model) {
    throw new Error("You must specify a model with your request.");
  }
  req.body.model = model;

  const requested = model;
  if (requested.startsWith("models/")) {
    req.body.model = requested.slice("models/".length);
  }

  if (requested.includes("gemini")) {
    return;
  }

  req.log.info({ requested }, "Reassigning model to gemini-1.5-pro-latest");
  req.body.model = "gemini-1.5-pro-latest";
}

export const googleAI = googleAIRouter;
