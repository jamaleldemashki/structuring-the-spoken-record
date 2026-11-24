// Password for the hosted app (Cloudflare Pages). Every request, the transcripts in data/
// included, has to carry the password first. The browser asks for it once and remembers it
// for the session.
//
// The password is not in the code. Set it once for the project:
//   npx wrangler@4.150.0 pages secret put SITE_PASSWORD --project-name <project>
// Any user name is accepted, only the password counts.

const encoder = new TextEncoder();

// Compares the hashes, so the time taken says nothing about how much of the password was right.
async function same(a, b) {
  const [x, y] = await Promise.all([a, b].map((s) => crypto.subtle.digest("SHA-256", encoder.encode(s))));
  const [u, v] = [new Uint8Array(x), new Uint8Array(y)];
  let diff = 0;
  for (let i = 0; i < u.length; i++) diff |= u[i] ^ v[i];
  return diff === 0;
}

function given(request) {
  const header = request.headers.get("Authorization") ?? "";
  if (!header.startsWith("Basic ")) return null;
  try {
    const decoded = atob(header.slice(6));
    return decoded.slice(decoded.indexOf(":") + 1);
  } catch {
    return null;
  }
}

export async function onRequest({ request, env, next }) {
  if (!env.SITE_PASSWORD) return new Response("The site password is not set.", { status: 503 });
  const password = given(request);
  if (password === null || !(await same(password, env.SITE_PASSWORD))) {
    return new Response("This archive is only open to study participants.", {
      status: 401,
      headers: { "WWW-Authenticate": 'Basic realm="Structuring the Spoken Record", charset="UTF-8"', "X-Robots-Tag": "noindex" },
    });
  }
  const response = await next();
  // Keep the site out of search engines even for visitors who have the password.
  const out = new Response(response.body, response);
  out.headers.set("X-Robots-Tag", "noindex, nofollow");
  return out;
}
