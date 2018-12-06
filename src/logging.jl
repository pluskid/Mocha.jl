using Memento
export m_debug, m_info, m_notice, m_warn, m_error

# NOTE: It isn't generally recommended to configure your logger at the package/library level.
push!(getlogger("Mocha"),
      DefaultHandler(stdout, DefaultFormatter("[{date} | {level} | {name}]: {msg}")))
setlevel!(getlogger("Mocha"), "info")
setpropagating!(getlogger("Mocha"), false)

function m_debug(msg :: AbstractString...)
  # TODO(pluskid): all the logging functions are now replaced with println due to some
  # mysterious segfault when calling memento functions. Revert to memento calls when 
  # this issue is resolved.
  println("[debug | Mocha]: " * prod(msg))
  # debug(getlogger("Mocha"), prod(msg))
end

function m_info(msg :: AbstractString...)
  println("[info | Mocha]: " * prod(msg))
  # info(getlogger("Mocha"), prod(msg))
end

function m_notice(msg :: AbstractString...)
  println("[notice | Mocha]: " * prod(msg))
  # notice(getlogger("Mocha"), prod(msg))
end

function m_warn(msg :: AbstractString...)
  println("[WARN | Mocha]: " * prod(msg))
  # warn(getlogger("Mocha"), prod(msg))
end

function m_error(msg :: AbstractString...)
  println("[ERROR | Mocha]: " * prod(msg))
  exit(1)
  # error(getlogger("Mocha"), prod(msg))
end
