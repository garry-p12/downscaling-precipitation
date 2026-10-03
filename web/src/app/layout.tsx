import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rain at One Kilometre",
  description:
    "IMERG satellite rainfall at 10 km, downscaled to 1 km and compared against AORC day by day.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
