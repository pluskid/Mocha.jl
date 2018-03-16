using Memento
export m_debug, m_info, m_notice, m_warn, m_error

# NOTE: It isn't generally recommended to configure your logger at the package/library level.
push!(getlogger(Mocha),
      DefaultHandler(STDOUT, DefaultFormatter("[{date} | {level} | {name}]: {msg}")))
setlevel!(getlogger(Mocha), "info")
setpropagating!(getlogger(Mocha), false)

function m_debug(msg :: AbstractString...)
  debug(getlogger(Mocha), prod(msg))
end

function m_info(msg :: AbstractString...)
  info(getlogger(Mocha), prod(msg))
end

function m_notice(msg :: AbstractString...)
  notice(getlogger(Mocha), prod(msg))
end

function m_warn(msg :: AbstractString...)
  warn(getlogger(Mocha), prod(msg))
end

function m_error(msg :: AbstractString...)
  error(getlogger(Mocha), prod(msg))
end
