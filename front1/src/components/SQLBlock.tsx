import { useState } from "react";
import { Check, ChevronDown, Code2, Copy } from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { copyText } from "@/lib/format";
import { useLanguage } from "@/context/LanguageContext";

interface Props {
  sql: string;
  defaultOpen?: boolean;
}

export default function SQLBlock({ sql, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const [copied, setCopied] = useState(false);
  const { t } = useLanguage();

  const onCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    const ok = await copyText(sql);
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    }
  };

  return (
    <div className="overflow-hidden rounded-xl border border-black/10 dark:border-white/10">
      <div className="flex items-center justify-between bg-black/[0.03] px-3 py-2 dark:bg-white/5">
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex items-center gap-2 text-xs font-medium text-black/70 transition hover:text-black dark:text-neutral-300 dark:hover:text-white"
        >
          <Code2 className="h-3.5 w-3.5" />
          {open ? t("hideSql") : t("viewSql")}
          <ChevronDown
            className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
          />
          <span className="ml-1 rounded bg-black/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-black/70 dark:bg-white/10 dark:text-neutral-300">
            PostgreSQL
          </span>
        </button>
        <button
          onClick={onCopy}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-black/60 transition hover:bg-black/10 hover:text-black dark:text-neutral-400 dark:hover:bg-white/10 dark:hover:text-white"
        >
          {copied ? (
            <Check className="h-3.5 w-3.5 text-emerald-500" />
          ) : (
            <Copy className="h-3.5 w-3.5" />
          )}
          {copied ? t("copied") : t("copy")}
        </button>
      </div>
      {open && (
        <div className="animate-fade-up text-[13px]">
          <SyntaxHighlighter
            language="sql"
            style={oneDark}
            customStyle={{
              margin: 0,
              borderRadius: 0,
              fontSize: 12.5,
              padding: "14px 16px",
              background: "#000000",
            }}
            wrapLongLines
          >
            {sql}
          </SyntaxHighlighter>
        </div>
      )}
    </div>
  );
}
