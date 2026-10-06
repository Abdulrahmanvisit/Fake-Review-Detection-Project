import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fake Review Detection | Review Analysis Workspace",
  description:
    "Analyse e-commerce reviews with aspect-based sentiment analysis and SVM classification.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
