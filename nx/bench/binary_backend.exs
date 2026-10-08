# Benchmark suite for Nx.BinaryBackend operations
# Run with: mix run bench/binary_backend.exs

defmodule Nx.Bench.Runner do
  def run(name, fun, opts \\ []) do
    warmup = Keyword.get(opts, :warmup, 5)
    runs = Keyword.get(opts, :runs, 20)

    # Warmup
    for _ <- 1..warmup, do: fun.()

    # Timed runs
    times =
      for _ <- 1..runs do
        {time, _result} = :timer.tc(fun)
        time / 1_000.0
      end

    mem = measure_memory(fun)

    avg = Enum.sum(times) / runs
    min = Enum.min(times)
    max = Enum.max(times)

    :io.format("~-30s ~8.3f ms  (min: ~6.3f ms, max: ~6.3f ms)  | ~9s~n", [
      name,
      avg,
      min,
      max,
      format_bytes(mem)
    ])
  end

  defp measure_memory(fun) do
    parent = self()

    spawn(fn ->
      :erlang.garbage_collect()
      {:memory, before_mem} = :erlang.process_info(self(), :memory)
      res = fun.()
      {:memory, after_mem} = :erlang.process_info(self(), :memory)
      send(parent, {:mem, max(0, after_mem - before_mem), res})
    end)

    receive do
      {:mem, bytes, _res} -> bytes
    end
  end

  defp format_bytes(bytes) do
    cond do
      bytes >= 1_048_576 -> :io_lib.format("~.2f MB", [bytes / 1_048_576]) |> to_string()
      bytes >= 1024 -> :io_lib.format("~.2f KB", [bytes / 1024]) |> to_string()
      true -> "#{bytes} B"
    end
  end
end

IO.puts("=================================================================")
IO.puts("Nx.BinaryBackend Benchmarks")
IO.puts("=================================================================")

size_1d = 100_000
t_f32 = Nx.iota({size_1d}, type: :f32)
t_f64 = Nx.iota({size_1d}, type: :f64)
t_s64 = Nx.iota({size_1d}, type: :s64)
t_s32 = Nx.iota({size_1d}, type: :s32)

m_size = 100
m_f32_a = Nx.iota({m_size, m_size}, type: :f32)
m_f32_b = Nx.iota({m_size, m_size}, type: :f32)
m_s64_a = Nx.iota({m_size, m_size}, type: :s64)
m_s64_b = Nx.iota({m_size, m_size}, type: :s64)

mat_2d = Nx.iota({500, 500}, type: :f32)

IO.puts("\n-- Reductions (1D, 100,000 elements) --")
Nx.Bench.Runner.run("sum (f32)", fn -> Nx.sum(t_f32) end)
Nx.Bench.Runner.run("sum (f64)", fn -> Nx.sum(t_f64) end)
Nx.Bench.Runner.run("sum (s64)", fn -> Nx.sum(t_s64) end)
Nx.Bench.Runner.run("sum (s32)", fn -> Nx.sum(t_s32) end)

Nx.Bench.Runner.run("product (f32)", fn -> Nx.product(t_f32) end)
Nx.Bench.Runner.run("product (s64)", fn -> Nx.product(t_s64) end)

Nx.Bench.Runner.run("reduce_max (f32)", fn -> Nx.reduce_max(t_f32) end)
Nx.Bench.Runner.run("reduce_max (s64)", fn -> Nx.reduce_max(t_s64) end)

Nx.Bench.Runner.run("reduce_min (f32)", fn -> Nx.reduce_min(t_f32) end)
Nx.Bench.Runner.run("reduce_min (s64)", fn -> Nx.reduce_min(t_s64) end)

IO.puts("\n-- Axis Reductions (2D, 500x500 matrix) --")
Nx.Bench.Runner.run("sum along axis 1 (f32)", fn -> Nx.sum(mat_2d, axes: [1]) end)
Nx.Bench.Runner.run("reduce_max along axis 1 (f32)", fn -> Nx.reduce_max(mat_2d, axes: [1]) end)

IO.puts("\n-- Matrix Multiplication (100x100 matrix) --")
Nx.Bench.Runner.run("dot 100x100 (f32)", fn -> Nx.dot(m_f32_a, m_f32_b) end, runs: 10)
Nx.Bench.Runner.run("dot 100x100 (s64)", fn -> Nx.dot(m_s64_a, m_s64_b) end, runs: 10)

IO.puts("=================================================================")
