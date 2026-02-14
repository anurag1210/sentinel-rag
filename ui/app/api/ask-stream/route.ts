export async function POST(req: Request) {
  const body = await req.json();

  const backendBase =
    process.env.NEXT_PUBLIC_API_BASE_URL ??
    "http://127.0.0.1:8000";

  const response = await fetch(`${backendBase}/ask-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  return new Response(response.body, {
    headers: {
      "Content-Type": "text/event-stream",
    },
  });
}