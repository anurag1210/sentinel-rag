"use client";

import { useMemo, useRef, useState } from "react";

type Msg = { role: "user" | "assistant"; text: string };

export default function Page() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Msg[]>([
    {
      role: "assistant",
      text: "Ask me anything about the uploaded book (RAG). I’ll answer using only the document context.",
    },
  ]);

  const endRef = useRef<HTMLDivElement | null>(null);

  // We store the assistant message index in a ref so it stays stable across async updates.
  const assistantIndexRef = useRef<number | null>(null);

  const canSend = useMemo(
    () => input.trim().length > 0 && !loading,
    [input, loading]
  );

  function appendToAssistant(delta: string) {
    setMessages((prev) => {
      const idx = assistantIndexRef.current;
      if (idx == null || idx < 0 || idx >= prev.length) return prev;

      const updated = [...prev];
      updated[idx] = { ...updated[idx], text: updated[idx].text + delta };
      return updated;
    });
  }

  async function onSend() {
    const q = input.trim();
    if (!q || loading) return;

    setInput("");
    setLoading(true);

    // Add user message
    setMessages((prev) => [...prev, { role: "user", text: q }]);

    // Add assistant placeholder message (empty for streaming)
    setMessages((prev) => {
      const idx = prev.length; // index of the new assistant message
      assistantIndexRef.current = idx;
      return [...prev, { role: "assistant", text: "" }];
    });

    try {
      const response = await fetch("/api/ask-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ question: q }),
      });

      if (!response.ok) {
        const txt = await response.text().catch(() => "");
        throw new Error(`HTTP ${response.status}. ${txt}`.trim());
      }

      if (!response.body) {
        throw new Error("No response body (stream missing).");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      // SSE parsing buffer
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete lines
        let lineEnd = buffer.indexOf("\n");
        while (lineEnd !== -1) {
          const rawLine = buffer.slice(0, lineEnd);
          buffer = buffer.slice(lineEnd + 1);

          // Remove trailing \r if present (Windows newlines)
          const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;

          // We only care about SSE "data:" lines
          if (line.startsWith("data:")) {
            // IMPORTANT: do NOT trim — preserve leading spaces and punctuation
            // "data:" can be "data: " or "data:" so slice(5) works
            let data = line.slice(5);
            if (data.startsWith(" ")) data = data.slice(1); // remove only the single separator space

            if (data === "[END]") {
              // End of stream signal from backend
              reader.cancel().catch(() => {});
              buffer = "";
              break;
            }

            // If backend sends an empty data line, treat as a newline
            if (data === "") {
              appendToAssistant("\n");
            } else {
              appendToAssistant(data);
            }
          }

          lineEnd = buffer.indexOf("\n");
        }

        setTimeout(
          () => endRef.current?.scrollIntoView({ behavior: "smooth" }),
          10
        );
      }
    } catch (e: any) {
      const msg = `Streaming failed. ${e?.message ?? ""}`.trim();
      appendToAssistant(`\n\n${msg}`);
    } finally {
      setLoading(false);
      setTimeout(
        () => endRef.current?.scrollIntoView({ behavior: "smooth" }),
        50
      );
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="mx-auto max-w-3xl px-4 py-10">
        <header className="mb-8">
          <div className="inline-flex items-center gap-2 rounded-full border border-neutral-800 bg-neutral-900 px-3 py-1 text-xs text-neutral-300">
            <span className="h-2 w-2 rounded-full bg-green-500" />
            Sentinel RAG • Local
          </div>
          <h1 className="mt-4 text-3xl font-semibold tracking-tight">
            Chat with your PDF
          </h1>
          <p className="mt-2 text-sm text-neutral-400">
            Answers are streamed from the backend (SSE). Sources can be shown later as citations.
          </p>
        </header>

        <section className="rounded-2xl border border-neutral-800 bg-neutral-900 shadow-sm">
          <div className="max-h-[62vh] overflow-y-auto px-4 py-4">
            <div className="space-y-4">
              {messages.map((m, idx) => (
                <div
                  key={idx}
                  className={`flex ${
                    m.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                      m.role === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-neutral-800 text-neutral-100"
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{m.text}</div>
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex justify-start">
                  <div className="max-w-[85%] rounded-2xl bg-neutral-800 px-4 py-3 text-sm text-neutral-200">
                    Thinking…
                  </div>
                </div>
              )}

              <div ref={endRef} />
            </div>
          </div>

          <div className="border-t border-neutral-800 p-4">
            <div className="flex gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
                className="min-h-[44px] w-full resize-none rounded-xl border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-500 focus:outline-none focus:ring-2 focus:ring-blue-600"
              />
              <button
                onClick={onSend}
                disabled={!canSend}
                className="rounded-xl bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
              >
                Send
              </button>
            </div>
          </div>
        </section>

        <footer className="mt-6 text-xs text-neutral-500">
          Next: stream structured events too (sources/metadata) instead of only raw text.
        </footer>
      </div>
    </main>
  );
}