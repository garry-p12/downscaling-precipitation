"use client";
export default function Info({ text }: { text: string }) {
  return <span className="info" title={text} role="img" aria-label={text}>i</span>;
}
