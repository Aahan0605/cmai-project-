export function Feedback({
  error,
  loading,
}: {
  error?: Error | null;
  loading: boolean;
}) {
  return error ? (
    <div className="notice" role="alert">
      Could not load results: {error.message}{" "}
      <button onClick={() => location.reload()}>Retry</button>
    </div>
  ) : loading ? (
    <p role="status">Loading recorded results…</p>
  ) : null;
}
