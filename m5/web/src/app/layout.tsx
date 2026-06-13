import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "MedSignal: Chẩn đoán xơ vữa động mạch cảnh",
  description:
    "Hỗ trợ chẩn đoán và phân tầng nguy cơ xơ vữa động mạch cảnh từ chỉ số lipid và ảnh siêu âm.",
  // Favicon lay tu file convention app/icon.jpg (khong can khai bao thu cong).
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full">
        <TooltipProvider>
          <div className="flex min-h-screen">
            <Sidebar />
            <div className="flex-1 overflow-x-hidden">{children}</div>
          </div>
        </TooltipProvider>
      </body>
    </html>
  );
}
