using Memento
export m_debug, m_info, m_notice, m_warn, m_error

remove_handler(get_logger(Mocha), "console")
add_handler(get_logger(Mocha),
            DefaultHandler(STDOUT,
                           DefaultFormatter("[{date} | {level} | {name}]: {msg}")))
set_level(get_logger(Mocha), "info")

function m_debug(msg :: AbstractString...)
  debug(get_logger(Mocha), prod(msg))
end

function m_info(msg :: AbstractString...)
  info(get_logger(Mocha), prod(msg))
end

function m_notice(msg :: AbstractString...)
  notice(get_logger(Mocha), prod(msg))
end

function m_warn(msg :: AbstractString...)
  warn(get_logger(Mocha), prod(msg))
end

function m_error(msg :: AbstractString...)
  error(get_logger(Mocha), prod(msg))
end
