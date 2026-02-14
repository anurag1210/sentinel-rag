import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const body = await req.json(); // { question: string }

  const backendBase =
    process.env.NEXT_PUBLIC_API_BASE_URL ??
    process.env.RAG_API_BASE ??
    "http://127.0.0.1:8000";

  const r = await fetch(`${backendBase}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const text = await r.text();

  return new NextResponse(text, {
    status: r.status,
    headers: { "Content-Type": "application/json" },
  });
}