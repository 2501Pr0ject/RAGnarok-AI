import "./globals.css";

export const metadata = {
  title: "RAGnarok Human Evaluation",
  description: "Open-source study: human annotation of RAG answers against their retrieved context.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <main>{children}</main>
      </body>
    </html>
  );
}
