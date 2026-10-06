import { NextResponse } from "next/server";

const BACKEND_URL =
  process.env.FLASK_API_URL ?? "http://127.0.0.1:5000/api/predict";

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  try {
    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      cache: "no-store",
      signal: AbortSignal.timeout(60_000),
    });

    let data: unknown;
    try {
      data = await response.json();
    } catch {
      return NextResponse.json(
        { error: "The analysis service returned an invalid response." },
        { status: 502 },
      );
    }

    return NextResponse.json(data, {
      status: response.status,
    });
  } catch {
    return NextResponse.json(
      {
        error:
          "Unable to connect to the analysis service. Please ensure it is running and try again.",
      },
      {
        status: 502,
      },
    );
  }
}