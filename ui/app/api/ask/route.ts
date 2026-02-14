import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const body = await req.json();

    const backendBase =
      process.env.NEXT_PUBLIC_API_BASE_URL ??
      "http://127.0.0.1:8000";

    const r = await fetch(`${backendBase}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await r.json();

    return NextResponse.json(data, { status: r.status });
  } catch (err: any) {
    return NextResponse.json(
      { error: "Proxy error", detail: err.message },
      { status: 500 }
    );
  }
}