import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The viewer is entirely client-side: it fetches JSON and PNG tiles from
  // /public and decodes them in the browser. There are no server components
  // doing data work, no route handlers and no middleware, so a static export
  // is the honest deployment target -- plain files on a CDN, no serverless
  // functions and no cold starts.
  output: "export",

  // `next/image` is unused (the maps are canvas), but this keeps the export
  // from failing if an <Image> is ever added without a loader.
  images: { unoptimized: true },

  // Emit /about/index.html rather than /about.html, which is what static hosts
  // expect when they serve a directory URL.
  trailingSlash: true,
};

export default nextConfig;
